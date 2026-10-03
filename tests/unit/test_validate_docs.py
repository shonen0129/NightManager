from scripts.ci import validate_docs


def test_iter_repository_paths_reads_path_tables_only(tmp_path):
    document = tmp_path / "architecture.md"
    document.write_text(
        "| Path | Purpose |\n"
        "|---|---|\n"
        "| src/leadlag/cli.py | CLI |\n"
        "\n"
        "| Topic | Detail |\n"
        "|---|---|\n"
        "| missing.py | prose |\n",
        encoding="utf-8",
    )

    assert [item[0] for item in validate_docs.iter_repository_paths(document)] == [
        "src/leadlag/cli.py"
    ]


def test_validate_rejects_missing_repository_path(tmp_path, monkeypatch, capsys):
    document = tmp_path / "architecture.md"
    document.write_text(
        "| Path | Purpose |\n|---|---|\n| missing/module.py | Stale |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(validate_docs, "REPOSITORY_ROOT", tmp_path)

    assert validate_docs.validate([document]) == 1
    assert "repository path does not exist: missing/module.py" in capsys.readouterr().out


def test_validate_accepts_current_repository_path(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/module.py").write_text("", encoding="utf-8")
    document = tmp_path / "architecture.md"
    document.write_text(
        "| Path | Purpose |\n|---|---|\n| src/module.py | Module |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(validate_docs, "REPOSITORY_ROOT", tmp_path)

    assert validate_docs.validate([document]) == 0
