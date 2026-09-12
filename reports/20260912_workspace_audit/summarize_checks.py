import ast
from collections import Counter
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
DIR=Path(__file__).resolve().parent
for name in ['inventory.json','ruff.json','reproductions.json']:
    p=DIR/name
    txt=p.read_text()
    if 'WATCHDOG:' in txt:
        txt=txt.split('WATCHDOG:')[0]+'\n'
        p.write_text(txt)
    json.loads(txt)
inv=json.loads((DIR/'inventory.json').read_text())
ruff=json.loads((DIR/'ruff.json').read_text())
out={'inventory':{k:{x:v[x] for x in ['files','python_files','python_lines']} for k,v in inv['inventory'].items()},'largest_production':inv['inventory']['src/leadlag']['largest_python'],'lint':{'total':len(ruff),'by_code':dict(Counter(r['code'] for r in ruff)),'by_root':dict(Counter(str(Path(r['filename']).relative_to(ROOT)).split('/')[0] for r in ruff)),'production':dict(Counter(r['code'] for r in ruff if '/src/leadlag/' in r['filename']))}}
out['fatal_lint']=[{'file':str(Path(r['filename']).relative_to(ROOT)),'line':r['location']['row'],'code':r['code'],'message':r['message']} for r in ruff if r['code'] in ['F821','F822','F823']]
# Discover production modules that are both a file and a package.
out['module_name_collisions']=[str(p.relative_to(ROOT)) for p in (ROOT/'src/leadlag').rglob('*.py') if p.with_suffix('').is_dir()]
experiments=list((ROOT/'src/research/scripts/experiments').glob('*.py'))
out['experiments']={'scripts':len(experiments),'registry_text_mentions':sum('record_backtest_experiment' in p.read_text() or 'ExperimentRegistry' in p.read_text() or 'record_simple_experiment' in p.read_text() for p in experiments),'default_registry_exists':(ROOT/'var/experiments/registry.jsonl').exists()}
# Current functions with unusually high branch counts, for prioritization only.
functions=[]
for p in (ROOT/'src/leadlag').rglob('*.py'):
    tree=ast.parse(p.read_text())
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            branches=sum(isinstance(x,(ast.If,ast.For,ast.While,ast.ExceptHandler,ast.IfExp)) for x in ast.walk(node))
            functions.append({'file':str(p.relative_to(ROOT)),'line':node.lineno,'name':node.name,'lines':node.end_lineno-node.lineno+1,'branches':branches})
out['branch_heavy_functions']=sorted(functions,key=lambda x:x['branches'],reverse=True)[:15]
print(json.dumps(out,ensure_ascii=False,indent=2))
