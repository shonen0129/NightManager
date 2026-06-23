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
from typing import Any

import requests

from leadlag.config import TachibanaApiConfig

logger = logging.getLogger(__name__)


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

    def __enter__(self) -> TachibanaClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.close()
        return False

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
            cipher = PKCS1_v1_5.new(key)
            sentinel = b"DECRYPT_FAIL"
            dec = cipher.decrypt(encrypted_data, sentinel)
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
        response.raise_for_status()

        result = response.json()
        # Check gateway-level errors first (e.g. invalid auth_id)
        p_errno = result.get("p_errno", "0")
        if p_errno != "0":
            err_text = result.get("p_err", "Tachibana gateway error")
            raise TachibanaApiError(
                f"Tachibana login failed (gateway code={p_errno}): {err_text}",
                endpoint="/auth/login",
                result_code=p_errno,
            )

        result_code = result.get("sResultCode", "-1")
        if result_code != "0":
            err_text = result.get("sResultText", "Unknown login error")
            raise TachibanaApiError(
                f"Tachibana login failed (code={result_code}): {err_text}",
                endpoint="/auth/login",
                result_code=result_code,
            )

        # Decrypt virtual URLs
        self.decrypted_urls = {}
        for url_key in ["sUrlRequest", "sUrlMaster", "sUrlPrice", "sUrlEvent"]:
            encrypted_val = result.get(url_key)
            if not encrypted_val:
                raise ValueError(f"Missing encrypted URL key '{url_key}' in login response.")
            decrypted_val = self._decrypt_virtual_url(encrypted_val)
            self.decrypted_urls[url_key] = decrypted_val

        self.logged_in = True
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
        result = response.json()

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
            result = response.json()

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
