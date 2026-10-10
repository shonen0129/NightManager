from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_close_cli_forwards_config(monkeypatch) -> None:
    import leadlag.cli as cli
    from leadlag.execution import close as close_module

    captured: dict[str, object] = {}

    def fake_run_close_positions_mode(**kwargs):
        captured.update(kwargs)
        return {"close_incomplete": False}

    monkeypatch.setattr(close_module, "run_close_positions_mode", fake_run_close_positions_mode)

    args = cli.setup_parser().parse_args(["close", "--config", "custom-production.yaml"])
    assert cli._handle_close(args) == 0
    assert captured["config_path"] == "custom-production.yaml"


def test_build_api_client_uses_supplied_config_and_explicit_overrides(monkeypatch) -> None:
    from leadlag.execution import broker_ops

    captured: dict[str, object] = {}

    class FakeClient:
        def health_check(self) -> bool:
            return True

    app_config = SimpleNamespace(
        broker_provider="tachibana",
        tachibana=SimpleNamespace(
            api_url="https://resolved.example",
            auth_id="resolved-auth",
            second_password="resolved-password",
            margin_trade_type=1,
            account_type=12,
            request_timeout=7,
            private_key_path="/resolved/key.pem",
        ),
        kabu=SimpleNamespace(
            api_url="http://unused.example",
            api_token="unused",
            api_password="unused",
            margin_trade_type=3,
            account_type=4,
            request_timeout=10,
        ),
    )

    def unexpected_loader():
        raise AssertionError("resolved config must not be reloaded")

    def fake_create_broker_from_args(**kwargs):
        captured.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(broker_ops, "load_config_from_yaml", unexpected_loader)
    monkeypatch.setattr(broker_ops, "create_broker_from_args", fake_create_broker_from_args)

    client = broker_ops.build_api_client(
        "https://override.example",
        "override-token",
        True,
        app_config=app_config,
    )

    assert isinstance(client, FakeClient)
    assert captured["provider"] == "tachibana"
    assert captured["dry_run"] is True
    assert captured["api_url"] == "https://override.example"
    assert captured["api_token"] == "override-token"
    assert captured["margin_trade_type"] == 1
    assert captured["account_type"] == 12
    assert captured["request_timeout"] == 7


def test_broker_factory_honors_resolved_provider_over_environment(monkeypatch) -> None:
    from leadlag.broker import factory

    captured: dict[str, object] = {}

    def fake_create_broker(config):
        captured["config"] = config
        return object()

    monkeypatch.setenv("BROKER_PROVIDER", "kabu")
    monkeypatch.setattr(factory, "create_broker", fake_create_broker)

    factory.create_broker_from_args(
        api_url="https://example.test",
        provider="tachibana",
        dry_run=False,
    )

    assert captured["config"].provider == "tachibana"


def test_decision_passes_same_resolved_config_to_broker(monkeypatch) -> None:
    from leadlag.execution import v2_bridge

    app_config = SimpleNamespace(
        v2=SimpleNamespace(ml_overlay_enabled=False),
        gap_distribution_dir="",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        v2_bridge,
        "load_config_from_yaml",
        lambda *_args, **_kwargs: app_config,
    )
    monkeypatch.setattr(
        v2_bridge,
        "_resolve_trade_date",
        lambda *_args, **_kwargs: "2026-10-09",
    )
    monkeypatch.setattr(v2_bridge, "_resolve_gap_dir", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(v2_bridge, "_load_df_exec", lambda *_args, **_kwargs: object())

    def stop_at_broker_boundary(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop at broker boundary")

    monkeypatch.setattr(v2_bridge, "build_api_client", stop_at_broker_boundary)

    with pytest.raises(RuntimeError, match="stop at broker boundary"):
        v2_bridge.run_v2_decision(
            config_path="unused.yaml",
            api_enable=True,
            api_dry_run=True,
        )

    assert captured["app_config"] is app_config


def test_close_uses_inherited_config_for_plan_broker_and_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from leadlag.execution import close as close_module

    root = Path(__file__).resolve().parents[2]
    custom_config = tmp_path / "custom-production.yaml"
    custom_config.write_text(
        json.dumps(
            {
                "__base__": str(root / "configs/production/production.yaml"),
                "overnight_alpha_long": 0.0,
                "overnight_alpha_short": 0.0,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("BROKER_PROVIDER", "tachibana")
    monkeypatch.setenv("TACHIBANA_AUTH_ID", "auth-for-test")
    monkeypatch.setenv("TACHIBANA_SECOND_PASSWORD", "second-for-test")
    monkeypatch.setenv("TACHIBANA_MARGIN_TRADE_TYPE", "1")
    monkeypatch.setenv("TACHIBANA_ACCOUNT_TYPE", "12")

    observed: dict[str, object] = {}

    class FakeClient:
        closed = False

        def close(self) -> None:
            self.closed = True

    client = FakeClient()

    def fake_build_api_client(*args, **kwargs):
        observed["broker_args"] = args
        observed["broker_kwargs"] = kwargs
        observed["app_config"] = kwargs["app_config"]
        return client

    def fake_close_all_positions(**kwargs):
        observed["close_kwargs"] = kwargs
        return {
            "close_incomplete": False,
            "filled_orders_count": 0,
            "close_results": [],
            "reconciliation_errors": [],
        }

    def fake_update_execution_manifest(*args, **kwargs):
        observed["manifest_args"] = args
        observed["manifest_kwargs"] = kwargs
        return str(tmp_path / "run_manifest.json")

    monkeypatch.setattr(close_module, "build_api_client", fake_build_api_client)
    monkeypatch.setattr(
        close_module,
        "build_output_dir",
        lambda *_args, **_kwargs: str(tmp_path),
    )
    monkeypatch.setattr(close_module, "ExecutionStateStore", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        close_module,
        "execution_lease",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(close_module, "close_all_positions", fake_close_all_positions)
    monkeypatch.setattr(
        close_module,
        "save_position_snapshot",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        close_module,
        "save_wallet_snapshot",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(close_module, "save_daily_journal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        close_module,
        "update_execution_manifest",
        fake_update_execution_manifest,
    )

    summary = close_module.run_close_positions_mode(
        output_root=str(tmp_path),
        run_tag="issue-61",
        api_url="https://cli-override.example",
        api_token="cli-token",
        api_dry_run=False,
        close_position_order=5,
        config_path=custom_config,
    )

    assert summary["close_incomplete"] is False
    app_config = observed["app_config"]
    assert app_config.broker_provider == "tachibana"
    assert app_config.strategy.overnight_alpha_long == 0.0
    assert app_config.strategy.overnight_alpha_short == 0.0
    assert app_config.tachibana.margin_trade_type == 1
    assert app_config.tachibana.account_type == 12

    broker_args = observed["broker_args"]
    assert broker_args == (
        "https://cli-override.example",
        "cli-token",
        False,
    )

    close_kwargs = observed["close_kwargs"]
    assert close_kwargs["overnight_alpha_long"] == 0.0
    assert close_kwargs["overnight_alpha_short"] == 0.0
    assert close_kwargs["margin_trade_type"] == 1
    assert close_kwargs["account_type"] == 12
    assert close_kwargs["close_position_order"] == 5
    assert close_kwargs["account_key"] == "tachibana:default"

    manifest = observed["manifest_kwargs"]["config"]
    assert manifest["resolved_hash"]
    assert manifest["non_secret_hash"]
    assert manifest["resolved_non_secret"]["broker_provider"] == "tachibana"
    assert manifest["resolved_non_secret"]["broker"]["margin_trade_type"] == 1
    assert manifest["resolved_non_secret"]["broker"]["account_type"] == 12
    assert manifest["resolved_non_secret"]["execution"]["overnight_alpha_long"] == 0.0
    assert manifest["resolved_non_secret"]["execution"]["overnight_alpha_short"] == 0.0
    serialized_manifest = json.dumps(manifest, sort_keys=True)
    assert "auth-for-test" not in serialized_manifest
    assert "second-for-test" not in serialized_manifest
    assert client.closed is True


def test_default_production_carry_is_unchanged(monkeypatch) -> None:
    from leadlag.execution.config import load_config_from_yaml

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("BROKER_PROVIDER", "kabu")

    config = load_config_from_yaml(root / "configs/production/production.yaml")

    assert config.strategy.overnight_alpha_long == 0.75
    assert config.strategy.overnight_alpha_short == 0.5
