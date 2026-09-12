"""Read existing stores through SQLite read-only mode; report aggregate coverage only."""
from pathlib import Path
import json
import pickle
import sqlite3
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from leadlag.data.cache_store import SqliteCacheStore
from leadlag.data.tickers import JP_TICKERS

# The deserializer is pure; skip __init__ to avoid schema/WAL writes.
decoder=object.__new__(SqliteCacheStore)
def connect(path):return sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5)
def value(db,key):
    row=db.execute('SELECT value FROM cache_store WHERE key=?',(key,)).fetchone()
    if row is None:return None
    blob=row[0]
    try:v=json.loads(blob)
    except (UnicodeDecodeError,json.JSONDecodeError):v=pickle.loads(blob)
    return decoder._restore_from_storage(v)

def df_summary(df):
    if df is None:return None
    return {'rows':len(df),'columns':len(df.columns),'start':str(df.index.min()),'end':str(df.index.max()),'duplicate_dates':int(df.index.duplicated().sum())}
out={}
with connect(ROOT/'var/market_data/df_exec.sqlite') as db:
    df=value(db,'df_exec');out['df_exec']=df_summary(df)
    out['df_exec']['provisional_rows']=int(df.get('is_provisional',pd.Series(dtype=float)).sum())
    out['df_exec']['baseline_rows']=int(((df.index>='2010-01-01')&(df.index<='2014-12-31')).sum())
    out['df_exec']['last_sig_date']=str(df['sig_date'].iloc[-1])
with connect(ROOT/'var/market_data/etf_prices.sqlite') as db:
    out['raw_keys']=[r[0] for r in db.execute('SELECT key FROM cache_store')]
    intraday=None
    for key in out['raw_keys']:
        if '5m' in key:
            intraday=value(db,key);out['intraday_5m']=df_summary(intraday)
    if intraday is not None:
        idx=intraday.index
        out['intraday_5m']['timezone']=str(idx.tz)
        bars=intraday[(idx.hour==9)&(idx.minute==10)].copy()
        dates=pd.to_datetime(bars.index.date)
        out['intraday_5m']['dates_with_910_bar']=len(set(dates))
        out['intraday_5m']['overlap_df_exec_days']=len(set(dates)&set(df.index))
        both=[]
        for tk in JP_TICKERS:
            try:valid=np.isfinite(bars[('High',tk)])&np.isfinite(bars[('Low',tk)])&(bars[('High',tk)]>0)&(bars[('Low',tk)]>0)
            except KeyError:valid=np.zeros(len(bars),dtype=bool)
            both.append(np.asarray(valid))
        valid_all=np.column_stack(both).all(axis=1)
        out['intraday_5m']['all_17_valid_910_days']=len(set(dates[valid_all])&set(df.index))
with connect(ROOT/'var/live/pipeline_data/gap_adjusted_distribution/gap_store.sqlite') as db:
    out['gap_store']={'matrix_coverage':db.execute('SELECT matrix_type,horizon,count(*),min(trade_date),max(trade_date) FROM gap_matrices GROUP BY matrix_type,horizon').fetchall()}
root=ROOT/'var/live/pipeline_data/gap_adjusted_distribution'
out['legacy_latest']={'exists':(root/'latest').exists(),'symlink_target':str((root/'latest').resolve().relative_to(ROOT)),'mu_count':len(list((root/'latest/matrices').glob('mu_gap_????????.npy')))}
for rel in ['full_history_diagnostics.csv','latest/portfolio_gap_distribution_diagnostics.csv']:
    p=root/rel
    if p.exists():
        diag=pd.read_csv(p)
        out[rel]={'rows':len(diag),'start':str(diag['trade_date'].min()),'end':str(diag['trade_date'].max()),'ir_columns':[c for c in diag.columns if 'ir' in c], 'duplicates':int(diag['trade_date'].duplicated().sum()),'sorted_dates':bool(diag['trade_date'].is_monotonic_increasing)}
for name in ['pit_binning.json','production_summary.csv']:
    p=ROOT/'var/live/production_residual_blpx'/name
    out[name]=json.loads(p.read_text()) if p.suffix=='.json' else pd.read_csv(p).to_dict('records')
print(json.dumps(out,ensure_ascii=False,indent=2,default=str))
