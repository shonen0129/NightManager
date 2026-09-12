"""Pure broker response mapping probes; no network connection is created."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from leadlag.broker.tachibana.client import TachibanaBrokerClient
from leadlag.execution.pricing import fetch_fill_prices

client=object.__new__(TachibanaBrokerClient)
responses={
 'filled':{'sOrderOrderSuryou':'100','sYakuzyouSuryou':'100','sOrderStatusCode':'10','sOrderStatus':'全部約定'},
 'cancelled':{'sOrderOrderSuryou':'100','sYakuzyouSuryou':'0','sOrderStatusCode':'7','sOrderStatus':'取消完了'},
 'rejected':{'sOrderOrderSuryou':'100','sYakuzyouSuryou':'0','sOrderStatusCode':'2','sOrderStatus':'受付エラー'},
}
out={}
for case,response in responses.items():
    client._client=SimpleNamespace(get_order_detail=lambda *args,response=response:response)
    out[case]={'fixture':response,'mapped_status':client.get_order_status('fake').value}
client.get_order_detail=Mock(side_effect=AssertionError('should not reach network'))
r={'order_id':'fake','ticker':'1617.T','status':'FILLED'}
fetch_fill_prices(client,[r],wait_seconds=0.)
out['filled_enrichment']={'result':r,'detail_called':client.get_order_detail.called}
print(json.dumps(out,ensure_ascii=False,indent=2))
