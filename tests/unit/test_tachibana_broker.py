"""tests/unit/test_tachibana_broker.py

Unit tests for TachibanaClient and TachibanaBrokerClient.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from leadlag.broker.base import BrokerConfig, WalletInfo
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
        api_url="https://demo-kabuka.e-shiten.jp/e_api_v4r10",
        auth_id="test_auth_id",
        private_key_path="dummy_key.pem",
        second_password="test_second_password",
        request_timeout=5,
    )


@pytest.fixture
def broker_config() -> BrokerConfig:
    return BrokerConfig(
        provider="tachibana",
        api_url="https://demo-kabuka.e-shiten.jp/e_api_v4r10",
        api_token="test_auth_id",
        api_password="test_second_password",
        request_timeout=5,
        extra={"private_key_path": "dummy_key.pem"},
    )


class TestTachibanaClient:
    def test_market_price_history_uses_v410_official_history_function(self, api_config):
        client = TachibanaClient(api_config)
        rows = [{"sDate": "20260928", "pDPP": "100.0000"}]
        with patch.object(
            client,
            "_request",
            return_value={"aCLMMfdsMarketPriceHistory": rows},
        ) as request:
            assert client.get_market_price_history("1617") == rows

        request.assert_called_once_with(
            "sUrlPrice",
            {
                "sCLMID": "CLMMfdsGetMarketPriceHistory",
                "sIssueCode": "1617",
                "sSizyouC": "00",
            },
        )

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
            mock_decrypt.side_effect = lambda val, **kwargs: f"https://decrypted-{val}.jp"
            client.login()

            assert client.logged_in is True
            assert client.decrypted_urls["sUrlRequest"] == "https://decrypted-encrypted_request_url.jp"
            assert client.decrypted_urls["sUrlPrice"] == "https://decrypted-encrypted_price_url.jp"
            assert client.p_no == 2

            diagnostics = client.last_login_diagnostics
            assert diagnostics is not None
            assert diagnostics["response_parsed"] is True
            assert diagnostics["p_errno"]["state"] == "absent"
            assert diagnostics["sResultCode"]["value"] == "0"
            assert diagnostics["login_success"] is True
            assert all(
                item["state"] == "nonempty"
                and item["decrypt_attempted"] is True
                and item["decrypt_succeeded"] is True
                for item in diagnostics["virtual_urls"].values()
            )
            serialized = json.dumps(diagnostics)
            assert "encrypted_request_url" not in serialized
            assert "https://decrypted-" not in serialized
            assert "test_auth_id" not in serialized

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
    def test_login_transport_failure_keeps_safe_unknown_diagnostics(self, mock_get, api_config):
        mock_get.side_effect = RuntimeError("network unavailable")

        client = TachibanaClient(api_config)
        with pytest.raises(RuntimeError, match="network unavailable"):
            client.login()

        diagnostics = client.last_login_diagnostics
        assert diagnostics is not None
        assert diagnostics["http_status"] is None
        assert diagnostics["response_parsed"] is False
        assert diagnostics["sResultCode"]["state"] == "unknown"
        assert diagnostics["sKinsyouhouMidokuFlg"]["state"] == "unknown"
        assert diagnostics["stopped_at"] == "transport_or_parse"
        assert all(
            item["state"] == "unknown"
            and item["decrypt_attempted"] is False
            and item["decrypt_succeeded"] is False
            for item in diagnostics["virtual_urls"].values()
        )
        serialized = json.dumps(diagnostics)
        assert "test_auth_id" not in serialized
        assert "test_second_password" not in serialized
        assert "dummy_key.pem" not in serialized

    @pytest.mark.parametrize(
        ("disclosure_flag", "expected_detail"),
        [
            ("1", "required disclosure documents are unread"),
            ("0", "sKinsyouhouMidokuFlg=0"),
        ],
    )
    @patch("requests.Session.get")
    def test_login_missing_virtual_url_reports_disclosure_status(
        self, mock_get, api_config, disclosure_flag, expected_detail
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sResultCode": "0",
            "sResultText": "",
            "sKinsyouhouMidokuFlg": disclosure_flag,
        }
        mock_get.return_value = mock_response

        client = TachibanaClient(api_config)
        with pytest.raises(ValueError) as exc_info:
            client.login()

        assert expected_detail in str(exc_info.value)
        assert client.logged_in is False

        diagnostics = client.last_login_diagnostics
        assert diagnostics is not None
        assert diagnostics["sKinsyouhouMidokuFlg"]["state"] == "value"
        assert diagnostics["sKinsyouhouMidokuFlg"]["value"] == disclosure_flag
        assert diagnostics["virtual_urls"]["sUrlRequest"]["state"] == "absent"
        assert diagnostics["virtual_urls"]["sUrlRequest"]["decrypt_attempted"] is False
        assert diagnostics["stopped_at"] == "missing_virtual_url:sUrlRequest"
        assert "test_auth_id" not in json.dumps(diagnostics)

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
            mock_decrypt.side_effect = lambda val, **kwargs: f"https://decrypted-{val}.jp"

            res = client._request("sUrlRequest", {"sCLMID": "CLMZanKaiSummary"})
            assert res["some_data"] == "ok"
            assert mock_get.call_count == 3


class TestTachibanaBrokerClient:
    @pytest.mark.parametrize(
        ("status_code", "filled", "ordered", "expected"),
        [
            ("7", 30, 100, OrderStatus.CANCELLED),
            ("7", 100, 100, OrderStatus.CANCELLED),
            ("9", 30, 100, OrderStatus.PARTIALLY_FILLED),
            ("10", 100, 100, OrderStatus.FILLED),
            ("11", 30, 100, OrderStatus.CANCELLED),
            ("12", 0, 100, OrderStatus.CANCELLED),
            ("19", 0, 100, OrderStatus.CANCELLED),
            ("5", 0, 100, OrderStatus.SUBMITTED),
            ("8", 30, 100, OrderStatus.PARTIALLY_FILLED),
            ("2", 100, 100, OrderStatus.FAILED),
            ("14", 100, 100, OrderStatus.FAILED),
            ("17", 100, 100, OrderStatus.FAILED),
            ("20", 100, 100, OrderStatus.FAILED),
            ("21", 100, 100, OrderStatus.FAILED),
        ],
    )
    def test_order_status_code_priority_and_request_failure_is_pending(
        self, broker_config, status_code, filled, ordered, expected
    ):
        client = create_broker(broker_config)
        client._client.get_order_detail = MagicMock(
            return_value={
                "sOrderStatusCode": status_code,
                "sYakuzyouSuryou": str(filled),
                "sOrderOrderSuryou": str(ordered),
            }
        )

        assert client.get_order_status("order-1") is expected

    @patch("requests.Session.get")
    def test_get_wallet_success(self, mock_get, broker_config):
        # Mock two API calls: CLMZanKaiSummary then CLMZanKaiSinyouSinkidateSyousai
        summary_response = MagicMock()
        summary_response.status_code = 200
        summary_response.json.return_value = {
            "sResultCode": "0",
            "sGenbutuKabuKaituke": "123456",
            "sSinyouSinkidate": "987654",
        }
        margin_response = MagicMock()
        margin_response.status_code = 200
        margin_response.json.return_value = {
            "sResultCode": "0",
            "sUkeireHosyoukin": "8631777",
            "sHosyoukinYoryoku": "8387119",
            "sHosyoukinRitu": "33",
        }
        mock_get.side_effect = [summary_response, margin_response]

        client = create_broker(broker_config)
        assert isinstance(client, TachibanaBrokerClient)

        client._client.logged_in = True
        client._client.decrypted_urls = {"sUrlRequest": "https://request-url.jp"}

        wallet = client.get_wallet()
        assert isinstance(wallet, WalletInfo)
        assert wallet.cash_available == 123456.0
        assert wallet.margin_available == 987654.0
        assert wallet.extra["ukeire_hosyoukin"] == 8631777.0
        assert wallet.extra["hosyoukin_yoryoku"] == 8387119.0
        assert wallet.extra["hosyoukin_ritu"] == 33.0

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

    @patch("requests.Session.get")
    def test_fetch_market_quotes_preserves_timestamp_and_lob(self, mock_get, broker_config):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sResultCode": "0",
            "aCLMMfdsMarketPrice": [
                {
                    "sIssueCode": "1617",
                    "pDPP": "1520.0000",
                    "pGAP1": "1521.0000",
                    "pGAV1": "100",
                    "pGBP1": "1519.0000",
                    "pGBV1": "80",
                }
            ],
        }
        mock_get.return_value = mock_response

        client = create_broker(broker_config)
        client._client.logged_in = True
        client._client.decrypted_urls = {"sUrlPrice": "https://price-url.jp"}

        quotes = client.fetch_market_quotes(
            ["1617.T"], observed_at="2026-09-23T09:10:01+09:00"
        )

        assert quotes[0]["ticker"] == "1617.T"
        assert quotes[0]["observed_at"] == "2026-09-23T09:10:01+09:00"
        assert quotes[0]["bid_price_1"] == 1519.0
        assert quotes[0]["ask_price_1"] == 1521.0
        assert quotes[0]["bid_size_1"] == 80.0
        assert quotes[0]["ask_size_1"] == 100.0
        assert quotes[0]["source"] == "tachibana:CLMMfdsGetMarketPrice"

    @patch("requests.Session.get")
    def test_fetch_market_quotes_does_not_invent_missing_ticker(self, mock_get, broker_config):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sResultCode": "0",
            "aCLMMfdsMarketPrice": [],
        }
        mock_get.return_value = mock_response

        client = create_broker(broker_config)
        client._client.logged_in = True
        client._client.decrypted_urls = {"sUrlPrice": "https://price-url.jp"}

        assert client.fetch_market_quotes(["1617.T"], allow_missing=True) == []

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
            "sEigyouDay": "20260717",
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
        assert res.eigyou_day == "20260717"

    @patch("requests.Session.get")
    def test_submit_orders_batch_rollback(self, mock_get, broker_config):
        # 1st order (BUY) succeeds, 2nd order (BUY) fails -> rollback triggers cancel on 1st order
        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.json.return_value = {
            "sResultCode": "0",
            "sOrderNumber": "1111",
            "sEigyouDay": "20260717",
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
        # Verify both sides are marked CANCELLED after market-neutrality rollback
        assert all(r.status == OrderStatus.CANCELLED for r in results)
        assert mock_get.call_count == 3
