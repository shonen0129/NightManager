"""tests/unit/test_tachibana_broker.py

Unit tests for TachibanaClient and TachibanaBrokerClient.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pytest
from leadlag.broker.base import BrokerConfig, Position, WalletInfo
from leadlag.broker.factory import create_broker
from leadlag.broker.tachibana.api import TachibanaApiError, TachibanaClient
from leadlag.broker.tachibana.client import TachibanaBrokerClient
from leadlag.config import TachibanaApiConfig
from leadlag.core.types import (
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
)


@pytest.fixture
def api_config() -> TachibanaApiConfig:
    return TachibanaApiConfig(
        api_url="https://demo-kabuka.e-shiten.jp/e_api_v4r9",
        auth_id="test_auth_id",
        private_key_path="dummy_key.pem",
        second_password="test_second_password",
        request_timeout=5,
    )


@pytest.fixture
def broker_config() -> BrokerConfig:
    return BrokerConfig(
        provider="tachibana",
        api_url="https://demo-kabuka.e-shiten.jp/e_api_v4r9",
        api_token="test_auth_id",
        api_password="test_second_password",
        request_timeout=5,
        extra={"private_key_path": "dummy_key.pem"},
    )


class TestTachibanaClient:
    @patch("requests.Session.get")
    def test_login_success(self, mock_get, api_config):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sResultCode": "0",
            "sResultText": "",
            "sUrlRequest": "encrypted_request_url",
            "sUrlMaster": "encrypted_master_url",
            "sUrlPrice": "encrypted_price_url",
            "sUrlEvent": "encrypted_event_url",
        }
        mock_get.return_value = mock_response

        client = TachibanaClient(api_config)

        # Mock decrypt helper to bypass file reading and RSA
        with patch.object(client, "_decrypt_virtual_url") as mock_decrypt:
            mock_decrypt.side_effect = lambda val: f"https://decrypted-{val}.jp"
            client.login()

            assert client.logged_in is True
            assert client.decrypted_urls["sUrlRequest"] == "https://decrypted-encrypted_request_url.jp"
            assert client.decrypted_urls["sUrlPrice"] == "https://decrypted-encrypted_price_url.jp"
            assert client.p_no == 2

    @patch("requests.Session.get")
    def test_login_failure(self, mock_get, api_config):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sResultCode": "10001",
            "sResultText": "Authentication failed",
        }
        mock_get.return_value = mock_response

        client = TachibanaClient(api_config)
        with pytest.raises(TachibanaApiError) as exc_info:
            client.login()

        assert "10001" in str(exc_info.value)
        assert client.logged_in is False

    @patch("requests.Session.get")
    def test_request_session_retry_on_timeout(self, mock_get, api_config):
        # 1st call returns session timeout, 2nd call (login) returns success, 3rd call (retry) returns success
        mock_res_timeout = MagicMock()
        mock_res_timeout.status_code = 200
        mock_res_timeout.json.return_value = {
            "sResultCode": "10099",
            "sResultText": "Session expired",
        }

        mock_res_login = MagicMock()
        mock_res_login.status_code = 200
        mock_res_login.json.return_value = {
            "sResultCode": "0",
            "sResultText": "",
            "sUrlRequest": "request_url",
            "sUrlMaster": "master_url",
            "sUrlPrice": "price_url",
            "sUrlEvent": "event_url",
        }

        mock_res_success = MagicMock()
        mock_res_success.status_code = 200
        mock_res_success.json.return_value = {
            "sResultCode": "0",
            "sResultText": "",
            "some_data": "ok",
        }

        mock_get.side_effect = [mock_res_timeout, mock_res_login, mock_res_success]

        client = TachibanaClient(api_config)
        client.logged_in = True
        client.decrypted_urls = {"sUrlRequest": "https://decrypted-request-url.jp"}

        with patch.object(client, "_decrypt_virtual_url") as mock_decrypt:
            mock_decrypt.side_effect = lambda val: f"https://decrypted-{val}.jp"

            res = client._request("sUrlRequest", {"sCLMID": "CLMZanKaiSummary"})
            assert res["some_data"] == "ok"
            assert mock_get.call_count == 3


class TestTachibanaBrokerClient:
    @patch("requests.Session.get")
    def test_get_wallet_success(self, mock_get, broker_config):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sResultCode": "0",
            "sGenbutuKabuKaituke": "123456",
            "sSinyouSinkidate": "987654",
        }
        mock_get.return_value = mock_response

        client = create_broker(broker_config)
        assert isinstance(client, TachibanaBrokerClient)

        client._client.logged_in = True
        client._client.decrypted_urls = {"sUrlRequest": "https://request-url.jp"}

        wallet = client.get_wallet()
        assert isinstance(wallet, WalletInfo)
        assert wallet.cash_available == 123456.0
        assert wallet.margin_available == 987654.0

    @patch("requests.Session.get")
    def test_fetch_open_prices(self, mock_get, broker_config):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sResultCode": "0",
            "aCLMMfdsMarketPrice": [
                {"sIssueCode": "1617", "pDOP": "1520.0000"},
                {"sIssueCode": "1618", "pDOP": "912.5000"},
            ],
        }
        mock_get.return_value = mock_response

        client = create_broker(broker_config)
        client._client.logged_in = True
        client._client.decrypted_urls = {"sUrlPrice": "https://price-url.jp"}

        opens = client.fetch_open_prices(["1617.T", "1618.T"])
        assert opens["1617.T"] == 1520.0
        assert opens["1618.T"] == 912.5

    @patch("yfinance.Ticker")
    def test_fetch_us_etf_returns_fallback(self, mock_ticker, broker_config):
        mock_hist = MagicMock()
        import pandas as pd
        mock_hist.history.return_value = pd.DataFrame(
            {"Close": [100.0, 102.5]},
            index=[pd.Timestamp("2026-06-19"), pd.Timestamp("2026-06-20")],
        )
        mock_ticker.return_value = mock_hist

        client = create_broker(broker_config)
        returns = client.fetch_us_etf_returns(["XLB"])
        assert returns["XLB"] == pytest.approx(0.025)

    @patch("requests.Session.get")
    def test_submit_order_market(self, mock_get, broker_config):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sResultCode": "0",
            "sOrderNumber": "887766",
        }
        mock_get.return_value = mock_response

        client = create_broker(broker_config)
        client._client.logged_in = True
        client._client.decrypted_urls = {"sUrlRequest": "https://request-url.jp"}

        order = OrderRequest(
            ticker="1617.T",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )

        res = client.submit_order(order)
        assert res.status == OrderStatus.SUBMITTED
        assert res.order_id == "887766"

    @patch("requests.Session.get")
    def test_submit_orders_batch_rollback(self, mock_get, broker_config):
        # 1st order (BUY) succeeds, 2nd order (BUY) fails -> rollback triggers cancel on 1st order
        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.json.return_value = {
            "sResultCode": "0",
            "sOrderNumber": "1111",
        }

        mock_fail = MagicMock()
        mock_fail.status_code = 200
        mock_fail.json.return_value = {
            "sResultCode": "11000",
            "sResultText": "Limit exceeded",
        }

        mock_cancel = MagicMock()
        mock_cancel.status_code = 200
        mock_cancel.json.return_value = {
            "sResultCode": "0",
        }

        mock_get.side_effect = [mock_success, mock_fail, mock_cancel]

        client = create_broker(broker_config)
        client._client.logged_in = True
        client._client.decrypted_urls = {"sUrlRequest": "https://request-url.jp"}

        orders = [
            OrderRequest(ticker="1617.T", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET),
            OrderRequest(ticker="1618.T", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET),
        ]

        results = client.submit_orders_batch(orders)
        # Verify both sides show failure due to rollback
        assert all(r.status == OrderStatus.FAILED for r in results)
        assert mock_get.call_count == 3
