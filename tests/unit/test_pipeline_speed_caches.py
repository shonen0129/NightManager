import json
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from leadlag.broker.tachibana import session_cache
from leadlag.broker.tachibana.api import TachibanaClient
from leadlag.data import cache as data_cache
from leadlag.execution import helpers


def test_decision_cache_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "decision_cache.npz"
    monkeypatch.setattr(data_cache, "decision_cache_path", lambda: str(path))
    index = pd.DatetimeIndex(["2026-07-28"], name="trade_date")
    df_exec = pd.DataFrame(
        {
            "sig_date": pd.to_datetime(["2026-07-27"]),
            "value": [1.0],
            "topix_night_return": [0.01],
        },
        index=index,
    )

    data_cache.save_decision_cache(df_exec)
    loaded = data_cache.load_decision_cache()

    pd.testing.assert_frame_equal(loaded, df_exec)


def test_open_cache_never_falls_back_to_another_date(tmp_path, monkeypatch):
    opens_dir = tmp_path / "opens"
    monkeypatch.setattr(
        session_cache,
        "_opens_cache_path",
        lambda trade_date: opens_dir / f"{trade_date}.csv",
    )
    opens_dir.mkdir()
    session_cache.save_open_prices_cache({"1617.T": 1000.0}, 2800.0, "20260728")

    assert session_cache.load_open_prices_cache("20260729") is None


def test_open_cache_rejects_non_positive_price(tmp_path, monkeypatch):
    opens_dir = tmp_path / "opens"
    monkeypatch.setattr(
        session_cache,
        "_opens_cache_path",
        lambda trade_date: opens_dir / f"{trade_date}.csv",
    )
    opens_dir.mkdir()
    pd.DataFrame(
        {"ticker": ["1617.T", "TOPIX"], "open_price": [0.0, 2800.0]}
    ).to_csv(opens_dir / "20260728.csv", index=False)

    assert session_cache.load_open_prices_cache("20260728") is None


def test_session_cache_is_private_and_stale_cache_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    monkeypatch.setattr(session_cache, "_session_cache_path", lambda: path)
    urls = {key: f"https://example.test/{key}" for key in session_cache._REQUIRED_URL_KEYS}
    session_cache.save_session_cache(
        {"decrypted_urls": urls, "p_no": 3, "logged_in": True}
    )

    assert os.stat(path).st_mode & 0o077 == 0
    assert session_cache.load_session_cache() is not None

    state = json.loads(path.read_text())
    state["saved_at"] = (
        datetime.now(UTC) - timedelta(minutes=16)
    ).isoformat()
    path.write_text(json.dumps(state))

    assert session_cache.load_session_cache() is None
    assert not path.exists()


def test_release_session_saves_without_logout(monkeypatch):
    client = TachibanaClient.__new__(TachibanaClient)
    client.logged_in = True
    client.decrypted_urls = {"sUrlRequest": "https://example.test/request"}
    client.p_no = 4
    client.session = SimpleNamespace(close=lambda: None)
    saved = []
    monkeypatch.setattr(session_cache, "save_session_cache", saved.append)

    client.release_session()

    assert saved == [
        {
            "decrypted_urls": {"sUrlRequest": "https://example.test/request"},
            "p_no": 4,
            "logged_in": True,
        }
    ]
    assert client.logged_in is True


def test_build_api_client_retries_after_invalid_restored_session(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.health_results = iter([False, True])
            self.discarded = False

        def restore_session(self):
            return True

        def health_check(self):
            return next(self.health_results)

        def discard_restored_session(self):
            self.discarded = True

    fake_client = FakeClient()
    app_cfg = SimpleNamespace(
        broker_provider="tachibana",
        tachibana=SimpleNamespace(
            api_url="https://example.test",
            auth_id="auth",
            second_password="password",
            margin_trade_type=3,
            account_type=4,
            request_timeout=5,
            private_key_path="/tmp/key.pem",
        ),
        kabu=None,
    )
    monkeypatch.setattr(helpers, "load_config_from_yaml", lambda: app_cfg)
    monkeypatch.setattr(helpers, "create_broker_from_args", lambda **kwargs: fake_client)

    result = helpers.build_api_client(None, None, False)

    assert result is fake_client
    assert fake_client.discarded is True
