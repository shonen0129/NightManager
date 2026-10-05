"""Tachibana Client API

Low-level client for Tachibana Securities e-Shiten API.
Handles PKI authentication, RSA decryption, session caching, and request formatting.
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.parse
from datetime import datetime
from typing import Any, cast

import requests

from leadlag.broker.tachibana import session_cache
from leadlag.config import TachibanaApiConfig

logger = logging.getLogger(__name__)


_LOGIN_URL_KEYS = ("sUrlRequest", "sUrlMaster", "sUrlPrice", "sUrlEvent")


def _diagnostic_scalar(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Describe a safe scalar field without defaulting a missing value."""
    if key not in payload:
        return {"state": "absent", "type": None, "value": None}
    value = payload[key]
    if value is None:
        return {"state": "null", "type": "NoneType", "value": None}
    if value == "":
        return {"state": "empty", "type": type(value).__name__, "value": ""}
    safe_value = value if isinstance(value, (str, int, float, bool)) else None
    return {"state": "value", "type": type(value).__name__, "value": safe_value}


def _diagnostic_url_state(payload: dict[str, Any], key: str) -> str:
    """Classify a virtual URL without recording ciphertext or decrypted values."""
    if key not in payload:
        return "absent"
    value = payload[key]
    if value is None:
        return "null"
    if value == "":
        return "empty"
    return "nonempty"


def _build_login_diagnostics(
    payload: dict[str, Any] | None,
    *,
    http_status: int | None,
    response_parsed: bool,
) -> dict[str, Any]:
    """Build an allowlisted login diagnostic record with no credentials or URL values."""
    if payload is None:
        scalar = {"state": "unknown", "type": None, "value": None}
        return {
            "http_status": http_status,
            "response_parsed": response_parsed,
            "sCLMID": dict(scalar),
            "p_errno": dict(scalar),
            "sResultCode": dict(scalar),
            "sKinsyouhouMidokuFlg": dict(scalar),
            "virtual_urls": {
                key: {
                    "state": "unknown",
                    "decrypt_attempted": False,
                    "decrypt_succeeded": False,
                }
                for key in _LOGIN_URL_KEYS
            },
            "login_success": False,
            "stopped_at": "transport_or_parse",
        }

    return {
        "http_status": http_status,
        "response_parsed": response_parsed,
        "sCLMID": _diagnostic_scalar(payload, "sCLMID"),
        "p_errno": _diagnostic_scalar(payload, "p_errno"),
        "sResultCode": _diagnostic_scalar(payload, "sResultCode"),
        "sKinsyouhouMidokuFlg": _diagnostic_scalar(payload, "sKinsyouhouMidokuFlg"),
        "virtual_urls": {
            key: {
                "state": _diagnostic_url_state(payload, key),
                "decrypt_attempted": False,
                "decrypt_succeeded": False,
            }
            for key in _LOGIN_URL_KEYS
        },
        "login_success": False,
        "stopped_at": None,
    }


class TachibanaApiError(Exception):
    """Raised when Tachibana API interactions fail."""

    def __init__(
        self,
        message: str,
        *,
        endpoint: str | None = None,
        result_code: str | None = None,
    ):
        self.endpoint = endpoint
        self.result_code = result_code
        super().__init__(message)


class TachibanaClient:
    """Low-level Tachibana Securities API Client.

    Communicates with e-Shiten API via GET/POST query parameters containing JSON request structures.
    Uses public/private key-pair to decrypt session virtual URLs.
    """

    def __init__(self, config: TachibanaApiConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.decrypted_urls: dict[str, str] = {}
        self.p_no = 1
        self.logged_in = False
        self.last_login_diagnostics: dict[str, Any] | None = None

    def __enter__(self) -> TachibanaClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close connection and logout if logged in."""
        if self.logged_in:
            try:
                self.logout()
            except Exception as e:
                logger.debug("Failed to logout during close: %s", e)
        self.session.close()

    def _get_timestamp(self) -> str:
        """Generate timestamp string in yyyy.mm.dd-hh:mn:ss.ttt format."""
        now = datetime.now()
        return now.strftime("%Y.%m.%d-%H:%M:%S") + f".{now.microsecond // 1000:03d}"

    def _decrypt_virtual_url(self, encrypted_b64: str) -> str:
        """Decrypt virtual URL base64 string using the PEM private key.

        Tries PKCS1_OAEP (SHA-256 / SHA-1) and PKCS1_v1_5.
        """
        from Crypto.Cipher import PKCS1_OAEP, PKCS1_v1_5
        from Crypto.Hash import SHA1, SHA256
        from Crypto.PublicKey import RSA

        try:
            with open(self.config.private_key_path, encoding="utf-8") as f:
                priv_key_pem = f.read()
        except Exception as e:
            raise ValueError(f"Failed to read private key from path '{self.config.private_key_path}': {e}")

        key = RSA.import_key(priv_key_pem)
        encrypted_data = base64.b64decode(encrypted_b64)

        # Try 1: PKCS1_OAEP with SHA-256
        try:
            cipher = PKCS1_OAEP.new(key, hashAlgo=SHA256)
            return cipher.decrypt(encrypted_data).decode("utf-8").strip()
        except Exception:
            pass

        # Try 2: PKCS1_OAEP with SHA-1 (default)
        try:
            cipher = PKCS1_OAEP.new(key, hashAlgo=SHA1)
            return cipher.decrypt(encrypted_data).decode("utf-8").strip()
        except Exception:
            pass

        # Try 3: PKCS1_v1_5
        try:
            v15_cipher = PKCS1_v1_5.new(key)
            sentinel = b"DECRYPT_FAIL"
            dec = v15_cipher.decrypt(encrypted_data, sentinel)
            if dec != sentinel:
                return dec.decode("utf-8").strip()
        except Exception:
            pass

        raise ValueError("Failed to decrypt virtual URL using all known RSA padding/hash algorithms.")

    def login(self) -> None:
        """Authenticate and decrypt session virtual URLs."""
        logger.info("[TachibanaAPI] Authenticating via PKI login...")
        p_sd_date = self._get_timestamp()
        payload = {
            "sCLMID": "CLMAuthLoginRequest",
            "sAuthId": self.config.auth_id,
            "p_no": str(self.p_no),
            "p_sd_date": p_sd_date,
            "sJsonOfmt": "4",
        }
        self.p_no += 1

        json_str = json.dumps(payload, separators=(",", ":"))
        url = f"{self.config.api_url.rstrip('/')}/auth/?{urllib.parse.quote(json_str)}"

        response = self.session.get(url, timeout=self.config.request_timeout)
        self.last_login_diagnostics = _build_login_diagnostics(
            None,
            http_status=response.status_code,
            response_parsed=False,
        )
        response.raise_for_status()

        try:
            result = cast(dict[str, Any], response.json())
        except Exception:
            self.last_login_diagnostics["stopped_at"] = "response_parse"
            raise
        self.last_login_diagnostics = _build_login_diagnostics(
            result,
            http_status=response.status_code,
            response_parsed=True,
        )
        # Check gateway-level errors first (e.g. invalid auth_id)
        p_errno = result.get("p_errno", "0")
        if p_errno != "0":
            self.last_login_diagnostics["stopped_at"] = "gateway_error"
            err_text = result.get("p_err", "Tachibana gateway error")
            raise TachibanaApiError(
                f"Tachibana login failed (gateway code={p_errno}): {err_text}",
                endpoint="/auth/login",
                result_code=p_errno,
            )

        result_code = result.get("sResultCode", "-1")
        if result_code != "0":
            self.last_login_diagnostics["stopped_at"] = "result_code_error"
            err_text = result.get("sResultText", "Unknown login error")
            raise TachibanaApiError(
                f"Tachibana login failed (code={result_code}): {err_text}",
                endpoint="/auth/login",
                result_code=result_code,
            )

        # Decrypt virtual URLs. Diagnostics record only presence/state and outcome.
        self.decrypted_urls = {}
        for url_key in _LOGIN_URL_KEYS:
            encrypted_val = result.get(url_key)
            url_diagnostics = self.last_login_diagnostics["virtual_urls"][url_key]
            if not encrypted_val:
                self.last_login_diagnostics["stopped_at"] = f"missing_virtual_url:{url_key}"
                disclosure_flag = result.get("sKinsyouhouMidokuFlg")
                if str(disclosure_flag) == "1":
                    raise ValueError(
                        "Tachibana API login succeeded but virtual URLs were not issued: "
                        "sKinsyouhouMidokuFlg=1 (required disclosure documents are unread; "
                        "review them in the standard e-Shiten website)."
                    )
                disclosure_status = (
                    f" (sKinsyouhouMidokuFlg={disclosure_flag})"
                    if disclosure_flag is not None
                    else ""
                )
                raise ValueError(
                    f"Missing encrypted URL key '{url_key}' in login response"
                    f"{disclosure_status}."
                )
            url_diagnostics["decrypt_attempted"] = True
            try:
                decrypted_val = self._decrypt_virtual_url(encrypted_val)
            except Exception:
                self.last_login_diagnostics["stopped_at"] = f"decrypt_virtual_url:{url_key}"
                raise
            url_diagnostics["decrypt_succeeded"] = True
            self.decrypted_urls[url_key] = decrypted_val

        self.logged_in = True
        self.last_login_diagnostics["login_success"] = True
        self.last_login_diagnostics["stopped_at"] = None
        logger.info("[TachibanaAPI] Login successful. Virtual URLs decrypted.")

    def logout(self) -> None:
        """Send logout request."""
        if not self.logged_in:
            return

        p_sd_date = self._get_timestamp()
        payload = {
            "sCLMID": "CLMAuthLogoutRequest",
            "p_no": str(self.p_no),
            "p_sd_date": p_sd_date,
            "sJsonOfmt": "4",
        }
        self.p_no += 1

        json_str = json.dumps(payload, separators=(",", ":"))
        url = f"{self.config.api_url.rstrip('/')}/auth/?{urllib.parse.quote(json_str)}"

        try:
            response = self.session.get(url, timeout=self.config.request_timeout)
            response.raise_for_status()
            logger.info("[TachibanaAPI] Logout successful")
        except Exception as e:
            logger.warning("[TachibanaAPI] Logout request failed: %s", e)
        finally:
            self.logged_in = False
            self.decrypted_urls = {}
            session_cache.clear_session_cache()

    def release_session(self) -> None:
        """Persist the authenticated server session and close only local transport."""
        self.save_session()
        self.session.close()

    def discard_restored_session(self) -> None:
        """Discard invalid restored state so the next request performs a fresh login."""
        self.logged_in = False
        self.decrypted_urls = {}
        session_cache.clear_session_cache()

    def save_session(self) -> None:
        """Persist decrypted virtual URLs and p_no to disk."""
        session_cache.save_session_cache({
            "decrypted_urls": self.decrypted_urls,
            "p_no": self.p_no,
            "logged_in": self.logged_in,
        })

    def restore_session(self, state: dict[str, Any]) -> bool:
        """Restore session state from a previously saved cache.

        Returns True if state was applied, False otherwise. The caller is
        responsible for validating the restored session with a health check.
        """
        if not state:
            return False
        try:
            self.decrypted_urls = dict(state.get("decrypted_urls", {}))
            self.p_no = int(state.get("p_no", 1))
            self.logged_in = bool(state.get("logged_in", False))
            logger.info("[TachibanaAPI] Restored session state (p_no=%d)", self.p_no)
            return True
        except Exception as e:
            logger.warning("[TachibanaAPI] Failed to restore session state: %s", e)
            return False

    def _request(
        self,
        url_key: str,
        payload: dict[str, Any],
        allow_relogin: bool = True,
    ) -> dict[str, Any]:
        """Perform request to the decrypted virtual URL with session renewal on timeout."""
        if not self.logged_in:
            self.login()

        virtual_url = self.decrypted_urls.get(url_key)
        if not virtual_url:
            raise ValueError(f"No decrypted virtual URL cached for '{url_key}'. Run login() first.")

        # Set tracking fields
        if "p_no" not in payload:
            payload["p_no"] = str(self.p_no)
            self.p_no += 1
        if "p_sd_date" not in payload:
            payload["p_sd_date"] = self._get_timestamp()
        if "sJsonOfmt" not in payload:
            payload["sJsonOfmt"] = "4"

        json_str = json.dumps(payload, separators=(",", ":"))
        full_url = f"{virtual_url.rstrip('/')}/?{urllib.parse.quote(json_str)}"

        response = self.session.get(full_url, timeout=self.config.request_timeout)
        response.raise_for_status()
        result = cast(dict[str, Any], response.json())

        # Check gateway-level errors first
        p_errno = result.get("p_errno", "0")
        if p_errno != "0":
            err_text = result.get("p_err", "Tachibana gateway error")
            raise TachibanaApiError(
                f"Tachibana request failed (gateway code={p_errno}): {err_text}",
                endpoint=payload.get("sCLMID"),
                result_code=p_errno,
            )

        # Check for session expired codes (10099 = セッションタイムアウト, 11991 = セッション情報レコードなし)
        result_code = result.get("sResultCode")
        if result_code in ("10099", "11991") and allow_relogin:
            logger.warning("[TachibanaAPI] Session expired (code=%s). Retrying after relogin...", result_code)
            self.login()
            # Retry request once with the new virtual URL and p_no
            payload["p_no"] = str(self.p_no)
            self.p_no += 1
            payload["p_sd_date"] = self._get_timestamp()
            new_virtual_url = self.decrypted_urls[url_key]
            new_json_str = json.dumps(payload, separators=(",", ":"))
            new_full_url = f"{new_virtual_url.rstrip('/')}/?{urllib.parse.quote(new_json_str)}"
            response = self.session.get(new_full_url, timeout=self.config.request_timeout)
            response.raise_for_status()
            result = cast(dict[str, Any], response.json())

            # Check gateway-level errors on retry
            p_errno = result.get("p_errno", "0")
            if p_errno != "0":
                err_text = result.get("p_err", "Tachibana gateway error")
                raise TachibanaApiError(
                    f"Tachibana request failed (gateway code={p_errno}): {err_text}",
                    endpoint=payload.get("sCLMID"),
                    result_code=p_errno,
                )

        # Generic error check (if sResultCode present and not 0)
        final_result_code = result.get("sResultCode", "0")
        if final_result_code != "0":
            err_text = result.get("sResultText", "Tachibana business logic error")
            raise TachibanaApiError(
                f"Tachibana request error (sCLMID={payload.get('sCLMID')}, code={final_result_code}): {err_text}",
                endpoint=payload.get("sCLMID"),
                result_code=final_result_code,
            )

        return result

    def get_wallet(self) -> dict[str, Any]:
        """Fetch balance summary details."""
        payload = {"sCLMID": "CLMZanKaiSummary"}
        return self._request("sUrlRequest", payload)

    def get_margin_detail(self, hituke_index: int = 0) -> dict[str, Any]:
        """Fetch margin detail (受入保証金, 保証金余力, etc.).

        Args:
            hituke_index: 0=当日, 1=翌営業日, ... 5=第6営業日
        """
        payload = {
            "sCLMID": "CLMZanKaiSinyouSinkidateSyousai",
            "sHitukeIndex": str(hituke_index),
        }
        return self._request("sUrlRequest", payload)

    def get_positions(self, ticker: str | None = None) -> list[dict[str, Any]]:
        """Fetch open credit positions."""
        payload = {
            "sCLMID": "CLMShinyouTategyokuList",
            "sIssueCode": ticker or "",
        }
        res = self._request("sUrlRequest", payload)
        # Check list elements
        return res.get("aShinyouTategyokuList") or []

    def get_price(self, tickers: list[str]) -> list[dict[str, Any]]:
        """Fetch current prices, open, previous close, and 5-level LOB for a list of tickers (max 120).

        LOB columns (ask side): pGAP1..5 (price), pGAV1..5 (size)
        LOB columns (bid side): pGBP1..5 (price), pGBV1..5 (size)
        """
        lob_columns = [
            "pDPP", "pPRP", "pDOP", "pDHP", "pDLP", "pDV",
            "pGAP1", "pGAP2", "pGAP3", "pGAP4", "pGAP5",
            "pGAV1", "pGAV2", "pGAV3", "pGAV4", "pGAV5",
            "pGBP1", "pGBP2", "pGBP3", "pGBP4", "pGBP5",
            "pGBV1", "pGBV2", "pGBV3", "pGBV4", "pGBV5",
        ]
        payload = {
            "sCLMID": "CLMMfdsGetMarketPrice",
            "sTargetIssueCode": ",".join(tickers),
            "sTargetColumn": ",".join(lob_columns),
        }
        res = self._request("sUrlPrice", payload)
        return res.get("aCLMMfdsMarketPrice") or []

    def get_market_price_history(self, ticker: str) -> list[dict[str, Any]]:
        """Fetch the broker's daily OHLC history for one issue code."""
        if not ticker or "," in ticker:
            raise ValueError("market-price history accepts exactly one ticker")
        payload = {
            "sCLMID": "CLMMfdsGetMarketPriceHistory",
            "sIssueCode": ticker,
            "sSizyouC": "00",
        }
        res = self._request("sUrlPrice", payload)
        return res.get("aCLMMfdsMarketPriceHistory") or []

    def send_order(
        self,
        ticker: str,
        side: str, # "1" for SELL, "3" for BUY
        quantity: int,
        order_price: str, # "0" for market/引成, or limit price
        condition: str = "0", # "0" for normal, "4" for 引け (used for CLO)
        genkin_shinyou: str = "0", # "2"=制度新規, "4"=制度返済, "6"=一般新規, "8"=一般返済
        account_type: str = "1", # "1"=特定, "3"=一般, "9"=法人
        is_close: bool = False,
    ) -> dict[str, Any]:
        """Submit trade order."""
        payload = {
            "sCLMID": "CLMKabuNewOrder",
            "sZyoutoekiKazeiC": account_type,
            "sIssueCode": ticker,
            "sSizyouC": "00", # 東証
            "sBaibaiKubun": side,
            "sCondition": condition,
            "sOrderPrice": order_price,
            "sOrderSuryou": str(quantity),
            "sGenkinShinyouKubun": genkin_shinyou,
            "sOrderExpireDay": "0", # 当日
            "sGyakusasiOrderType": "0", # 通常
            "sGyakusasiZyouken": "0",
            "sGyakusasiPrice": "*",
            "sTatebiType": "2" if is_close else "*", # 2=建日順 (when closing)
            "sTategyokuZyoutoekiKazeiC": "*",
            "sSecondPassword": self.config.second_password,
        }
        return self._request("sUrlRequest", payload)

    def cancel_order(self, order_number: str, eigyou_day: str) -> dict[str, Any]:
        """Cancel an open order."""
        payload = {
            "sCLMID": "CLMKabuCancelOrder",
            "sOrderNumber": order_number,
            "sEigyouDay": eigyou_day,
            "sSecondPassword": self.config.second_password,
        }
        return self._request("sUrlRequest", payload)

    def get_order_list(self) -> list[dict[str, Any]]:
        """Fetch today's orders."""
        payload = {
            "sCLMID": "CLMOrderList",
            "sIssueCode": "",
            "sSikkouDay": "",
            "sOrderSyoukaiStatus": "",
        }
        res = self._request("sUrlRequest", payload)
        return res.get("aOrderList") or []

    def get_order_detail(self, order_number: str, eigyou_day: str) -> dict[str, Any]:
        """Fetch order detail with fill information via CLMOrderListDetail.

        Args:
            order_number: Order number from CLMKabuNewOrder response
            eigyou_day: Business day (YYYYMMDD) from CLMKabuNewOrder response

        Returns:
            Dict with fill price (sYakuzyouPrice), fill quantity (sYakuzyouSuryou),
            fill date (sYakuzyouDate), and aYakuzyouSikkouList for partial fills.
        """
        payload = {
            "sCLMID": "CLMOrderListDetail",
            "sOrderNumber": order_number,
            "sEigyouDay": eigyou_day,
        }
        return self._request("sUrlRequest", payload)
