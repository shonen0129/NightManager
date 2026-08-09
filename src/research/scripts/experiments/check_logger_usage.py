import ast
from pathlib import Path

project = Path('/Users/shonen/日米ラグ')
errors = []

for py_file in project.rglob('*.py'):
    rel = py_file.relative_to(project)
    if '.venv' in str(rel):
        continue
    try:
        source = py_file.read_text(encoding='utf-8')
    except Exception:
        continue
    try:
        tree = ast.parse(source)
    except SyntaxError:
        continue

    logger_defined = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'logger':
                    logger_defined = True

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == 'logger'
        ):
            if not logger_defined:
                errors.append(f"{rel}: logger.{node.attr}")

if errors:
    print("Potential logger NameError:")
    for e in errors[:50]:
        print(e)
else:
    print("No module-level logger issues found.")
