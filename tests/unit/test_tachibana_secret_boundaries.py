"""Synthetic-only regressions for API diagnostics, RSA and broker outputs."""

from __future__ import annotations

import base64
import json
import logging
import traceback
from unittest.mock import patch

import pytest
import requests
from Crypto.Cipher import PKCS1_OAEP, PKCS1_v1_5
from Crypto.Hash import SHA1, SHA256
from Crypto.PublicKey import RSA

from leadlag.broker.base import BrokerConfig
from leadlag.broker.tachibana.api import TachibanaApiError, TachibanaClient
from leadlag.broker.tachibana.client import TachibanaBrokerClient
from leadlag.config import TachibanaApiConfig
from leadlag.core.types import OrderRequest, OrderSide, OrderStatus, OrderType

SECRET = "SYNTHETIC_CREDENTIAL_34"
URL = f"https://invalid.test/{SECRET}/?token={SECRET}"
ALGORITHMS = ["oaep_sha256", "oaep_sha1", "pkcs1_v1_5"]


@pytest.fixture(scope="module")
def rsa_key():
    return RSA.generate(2048)


@pytest.fixture
def rsa_client(tmp_path, rsa_key):
    key_path = tmp_path / f"{SECRET}.pem"
    key_path.write_bytes(rsa_key.export_key())
    client = TachibanaClient(TachibanaApiConfig(private_key_path=str(key_path), auth_id=SECRET))
    yield client
    client.session.close()


def encrypt(key, algorithm, plaintext):
    public = key.public_key()
    if algorithm == "pkcs1_v1_5":
        cipher = PKCS1_v1_5.new(public)
    else:
        cipher = PKCS1_OAEP.new(public, hashAlgo=SHA256 if algorithm == "oaep_sha256" else SHA1)
    return base64.b64encode(cipher.encrypt(plaintext)).decode("ascii")


def response(payload=None, status=200):
    result = requests.Response()
    result.status_code = status
    result.url = URL
    result._content = json.dumps(payload).encode() if payload is not None else SECRET.encode()
    return result


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_real_rsa_login_fallback_records_only_algorithm_and_stage(
    rsa_client, rsa_key, algorithm, caplog
):
    encrypted = encrypt(rsa_key, algorithm, URL.encode())
    payload = {key: encrypted for key in ("sUrlRequest", "sUrlMaster", "sUrlPrice", "sUrlEvent")}
    payload.update(sResultCode="0", sCLMID="CLMAuthLoginAck")
    with (
        patch.object(rsa_client.session, "get", return_value=response(payload)),
        caplog.at_level(logging.DEBUG),
    ):
        rsa_client.login()
    assert rsa_client.logged_in
    assert rsa_client.last_login_diagnostics["sCLMID"]["value"] == "CLMAuthLoginAck"
    assert set(rsa_client.decrypted_urls.values()) == {URL}
    for diagnostic in rsa_client.last_login_diagnostics["virtual_urls"].values():
        assert diagnostic["decrypt_algorithm"] == algorithm
        assert diagnostic["decrypt_failure_stage"] is None
        assert diagnostic["decrypt_attempts"] == [
            {"algorithm": item, "outcome": "succeeded" if item == algorithm else "decrypt_failed"}
            for item in ALGORITHMS[: ALGORITHMS.index(algorithm) + 1]
        ]
    observed = json.dumps(rsa_client.last_login_diagnostics) + caplog.text
    for secret in (SECRET, URL, encrypted, rsa_key.export_key().decode()):
        assert secret not in observed
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)


@pytest.mark.parametrize(
    "failure",
    ["all_algorithms", "invalid_utf8", "key_read", "key_import", "public_key", "ciphertext_decode"],
)
def test_rsa_expected_failures_have_safe_valueerror_contract(rsa_client, rsa_key, failure):
    encrypted = base64.b64encode(bytes(rsa_key.size_in_bytes())).decode()
    stage = failure
    if failure == "invalid_utf8":
        encrypted = encrypt(rsa_key, "oaep_sha256", b"\xff" + SECRET.encode())
        stage = "all_algorithms"
    elif failure == "key_read":
        rsa_client.config = rsa_client.config.model_copy(
            update={"private_key_path": f"/missing/{SECRET}.pem"}
        )
    elif failure in ("key_import", "public_key"):
        from pathlib import Path

        Path(rsa_client.config.private_key_path).write_bytes(
            rsa_key.public_key().export_key() if failure == "public_key" else SECRET.encode()
        )
        stage = "key_import"
    elif failure == "ciphertext_decode":
        encrypted = SECRET + "é"
    diagnostics = {}
    with pytest.raises(ValueError) as captured:
        rsa_client._decrypt_virtual_url(encrypted, diagnostics=diagnostics)
    assert diagnostics["decrypt_failure_stage"] == stage
    assert diagnostics["decrypt_algorithm"] is None
    if stage == "all_algorithms":
        assert (
            str(captured.value)
            == "Failed to decrypt virtual URL using all known RSA padding/hash algorithms."
        )
        assert [item["algorithm"] for item in diagnostics["decrypt_attempts"]] == ALGORITHMS
        if failure == "invalid_utf8":
            assert diagnostics["decrypt_attempts"][0]["outcome"] == "invalid_utf8"
    observed = json.dumps(diagnostics) + "".join(traceback.format_exception(captured.value))
    assert SECRET not in observed
    assert encrypted not in observed
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__


def test_rsa_does_not_swallow_unexpected_programming_error(rsa_client):
    with (
        patch("Crypto.Cipher.PKCS1_OAEP.new", side_effect=RuntimeError("implementation bug")),
        patch("Crypto.Cipher.PKCS1_v1_5.new") as fallback,
    ):
        with pytest.raises(RuntimeError, match="implementation bug"):
            rsa_client._decrypt_virtual_url("YQ==")
    fallback.assert_not_called()


def test_login_decrypt_failure_records_stage_without_exposing_key(rsa_client):
    rsa_client.config = rsa_client.config.model_copy(
        update={"private_key_path": f"/missing/{SECRET}.pem"}
    )
    payload = {"sResultCode": "0", "sUrlRequest": SECRET}
    with (
        patch.object(rsa_client.session, "get", return_value=response(payload)),
        pytest.raises(ValueError) as captured,
    ):
        rsa_client.login()
    diagnostics = rsa_client.last_login_diagnostics
    assert diagnostics["stopped_at"] == "decrypt_virtual_url:sUrlRequest"
    assert diagnostics["virtual_urls"]["sUrlRequest"]["decrypt_failure_stage"] == "key_read"
    assert not rsa_client.logged_in
    assert SECRET not in json.dumps(diagnostics) + "".join(
        traceback.format_exception(captured.value)
    )


@pytest.mark.parametrize("field", ["sCLMID", "p_errno", "sResultCode", "sKinsyouhouMidokuFlg"])
@pytest.mark.parametrize("raw_value", [SECRET, "12345678901234567890", {"payload": SECRET}])
def test_login_diagnostics_reject_arbitrary_values_in_allowlisted_fields(field, raw_value):
    client = TachibanaClient(TachibanaApiConfig(auth_id=SECRET))
    payload = {"sResultCode": "0", field: raw_value}
    with (
        patch.object(client.session, "get", return_value=response(payload)),
        pytest.raises((TachibanaApiError, ValueError)) as captured,
    ):
        client.login()
    observed = json.dumps(client.last_login_diagnostics) + "".join(
        traceback.format_exception(captured.value)
    )
    assert SECRET not in observed
    assert "12345678901234567890" not in observed
    assert client.last_login_diagnostics[field]["value"] is None
    client.session.close()


@pytest.mark.parametrize("operation", ["login", "order", "health"])
@pytest.mark.parametrize("failure", ["http", "transport", "parse", "server"])
def test_actual_broker_failures_do_not_leak_into_logs_or_order_summary(operation, failure, caplog):
    broker = TachibanaBrokerClient(
        BrokerConfig(provider="tachibana", api_token=SECRET, api_password=SECRET)
    )
    if operation != "login":
        broker._client.logged_in = True
        broker._client.decrypted_urls = {"sUrlRequest": URL, "sUrlPrice": URL}
    if failure == "http":
        fake_response = response(status=404)
    elif failure == "parse":
        fake_response = response()
    else:
        fake_response = response({"p_errno": "0", "sResultCode": SECRET, "sResultText": SECRET})
    error = requests.ConnectionError(
        URL, request=requests.Request("GET", URL).prepare(), response=fake_response
    )
    with (
        patch.object(
            broker._client.session,
            "get",
            side_effect=error if failure == "transport" else None,
            return_value=fake_response,
        ) as get,
        caplog.at_level(logging.DEBUG),
    ):
        if operation == "order":
            order = OrderRequest(
                ticker="1617.T", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET
            )
            result = broker.submit_order(order)
            assert result.status is OrderStatus.FAILED
            summary = result.message
        else:
            assert broker.health_check() is False
            summary = json.dumps(broker.last_login_diagnostics)
    assert get.call_count == 1
    # The secret really did traverse the request boundary, before being discarded.
    from urllib.parse import unquote

    assert SECRET in unquote(get.call_args.args[0])
    assert SECRET not in caplog.text + summary
    assert URL not in caplog.text + summary
    broker._client.session.close()
