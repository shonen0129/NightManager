#!/usr/bin/env python
"""Test Tachibana Securities API connection using .env credentials."""

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from leadlag.broker.tachibana.api import TachibanaClient, TachibanaApiError
from leadlag.config import TachibanaApiConfig

cfg = TachibanaApiConfig(
    api_url=os.environ.get("TACHIBANA_API_URL", ""),
    auth_id=os.environ.get("TACHIBANA_AUTH_ID", ""),
    private_key_path=os.environ.get("TACHIBANA_PRIVATE_KEY_PATH", ""),
    second_password=os.environ.get("TACHIBANA_SECOND_PASSWORD", ""),
    request_timeout=int(os.environ.get("TACHIBANA_REQUEST_TIMEOUT", "15")),
)

print(f"API URL: {cfg.api_url}")
print(f"Auth ID: {cfg.auth_id[:10]}...")
print(f"Key path: {cfg.private_key_path}")
print(f"Key exists: {os.path.exists(cfg.private_key_path)}")
print()

client = TachibanaClient(cfg)
try:
    print("=== Login ===")
    client.login()
    print(f"Login OK: {client.logged_in}")
    for k, v in client.decrypted_urls.items():
        print(f"  {k}: {v[:50]}...")

    print()
    print("=== Price query (1617) ===")
    res = client.get_price(["1617"])
    print(f"Result: {res}")

    print()
    print("=== Wallet query ===")
    res2 = client.get_wallet()
    for k, v in res2.items():
        print(f"  {k}: {v}")

    print()
    print("ALL TESTS PASSED")

except TachibanaApiError as e:
    print(f"API Error: {e}")
    print(f"  endpoint: {e.endpoint}")
    print(f"  result_code: {e.result_code}")
except Exception:
    traceback.print_exc()
finally:
    client.close()
    print()
    print("Connection closed.")
