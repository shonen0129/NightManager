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


def test_current_references_detect_stale_paths_in_prose_and_fenced_commands(tmp_path, monkeypatch, capsys):
    document = tmp_path / "operations.md"
    document.write_text(
        "Use `src/missing.py` and [tools/missing.py](tools/missing.py), then run:\n"
        "```bash\npython tools/missing.py\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(validate_docs, "REPOSITORY_ROOT", tmp_path)
    assert validate_docs.validate_current_references([document]) == 1
    output = capsys.readouterr().out
    assert "src/missing.py" in output
    assert "tools/missing.py" in output


def test_current_python_symbols_are_checked_without_import_side_effects(tmp_path, monkeypatch, capsys):
    (tmp_path / "src").mkdir()
    module = tmp_path / "src/model.py"
    module.write_text(
        "raise RuntimeError('must not import documentation targets')\n"
        "class Model:\n    def decide(self): pass\n",
        encoding="utf-8",
    )
    document = tmp_path / "operations.md"
    document.write_text("`src/model.py::Model.decide()`", encoding="utf-8")
    monkeypatch.setattr(validate_docs, "REPOSITORY_ROOT", tmp_path)
    assert validate_docs.validate_current_references([document]) == 0
    module.write_text("class Model:\n    def renamed(self): pass\n", encoding="utf-8")
    assert validate_docs.validate_current_references([document]) == 1
    assert "Model.decide" in capsys.readouterr().out


def test_historical_material_is_not_forced_to_match_current_api(tmp_path, monkeypatch):
    document = tmp_path / "operations.md"
    document.write_text(
        "<!-- docs:historical -->\n`src/old.py::OldModel`\n"
        "<!-- docs:current -->\n`src/current.py`\n",
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src/current.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(validate_docs, "REPOSITORY_ROOT", tmp_path)
    assert validate_docs.validate_current_references([document]) == 0
    history = tmp_path / "report.md"
    history.write_text("Historical `src/deleted.py::OldModel`", encoding="utf-8")
    assert validate_docs.validate([history]) == 0


def test_static_references_skip_runtime_templates_but_reject_root_escape(tmp_path, monkeypatch):
    document = tmp_path / "operations.md"
    document.write_text(
        "`var/live/generated.json` `src/research/<experiment>.py` "
        "`tests/test_*.py` `docs/YYYYMMDD_report.md`\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(validate_docs, "REPOSITORY_ROOT", tmp_path)
    assert validate_docs.validate_current_references([document]) == 0
    document.write_text("`src/../../outside.py`", encoding="utf-8")
    assert validate_docs.validate_current_references([document]) == 1


def test_current_discovery_includes_new_skills_but_not_reports(tmp_path, monkeypatch):
    skill = tmp_path / ".agents/skills/new-skill/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# New skill", encoding="utf-8")
    report = tmp_path / "reports/old.md"
    report.parent.mkdir()
    report.write_text("old", encoding="utf-8")
    monkeypatch.setattr(validate_docs, "REPOSITORY_ROOT", tmp_path)
    paths = validate_docs.current_documents()
    assert skill in paths
    assert report not in paths


def test_current_repository_documentation_references():
    paths = validate_docs.current_documents()
    assert validate_docs.validate(paths) == 0
    assert validate_docs.validate_current_references(paths) == 0
