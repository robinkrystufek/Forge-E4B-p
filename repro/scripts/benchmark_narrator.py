import argparse, dataclasses, hashlib, json, os, time
from pathlib import Path
import litert_lm

MODEL=Path('')
EXPECTED=''
PROMPT=Path(__file__).with_name('benchmark.narrator.v1.txt').read_text(encoding='utf-8')
CACHE=Path('')

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--backend',choices=['cpu','gpu'],required=True); ap.add_argument('--mtp',choices=['off','on'],required=True); ap.add_argument('--runs',type=int,default=3); ap.add_argument('--output',required=True); a=ap.parse_args()
 model=a.model.resolve(); cache=a.cache.resolve();
 with model.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
 if EXPECTED and digest!=EXPECTED: raise SystemExit(f'P10 SHA mismatch: {digest}')
 cache.mkdir(parents=True,exist_ok=True)\n dll_dir=os.add_dll_directory(str(Path(litert_lm.__file__).resolve().parent))
 backend=litert_lm.Backend.GPU() if a.backend=='gpu' else litert_lm.Backend.CPU()
 mode=litert_lm.SpeculativeDecodingMode.ON if a.mtp=='on' else litert_lm.SpeculativeDecodingMode.OFF
 cfg=litert_lm.SpeculativeDecodingConfig(mode=mode)
 engine=litert_lm.Engine(str(model),backend=backend,max_num_tokens=4096,cache_dir=str(cache),enable_benchmark=True,speculative_decoding_config=cfg)
 rows=[]
 try:
  for i in range(a.runs):
   with engine.create_conversation(sampler_config=litert_lm.SamplerConfig(top_k=64,top_p=0.95,temperature=1.0),thinking_config=litert_lm.ThinkingConfig(enable_thinking=False,thinking_token_budget=0),max_output_tokens=500) as conv:
    start=time.perf_counter(); reply=conv.send_message(PROMPT); elapsed=time.perf_counter()-start; info=conv.get_benchmark_info()
    content=getattr(reply,'content','')
    if not isinstance(content,str): content=str(content)
    row={'iteration':i+1,'total_seconds':elapsed,'ttft_seconds':info.time_to_first_token_in_second,'input_tokens':info.last_prefill_token_count,'output_tokens':info.last_decode_token_count,'prefill_tokens_per_second':info.last_prefill_tokens_per_second,'decode_tokens_per_second':info.last_decode_tokens_per_second,'valid_long_output':info.last_decode_token_count>=200,'output_preview':content[:120]}
    rows.append(row); print(json.dumps(row),flush=True)
 finally: engine.close()
 result={'package_sha256':digest,'runtime':'LiteRT-LM Python conversation API 0.19.2','backend':a.backend,'device':'RTX 4090 / WebGPU-Dawn D3D12' if a.backend=='gpu' else 'CPU','mtp':a.mtp,'prompt_id':'Forge-GM benchmark.narrator v1','prompt_source':'repro/fixtures/benchmark.narrator.v1.txt','prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),'sampler':{'top_k':64,'top_p':0.95,'temperature':1.0},'thinking':False,'output_token_limit':500,'max_context_tokens':4096,'minimum_valid_output_tokens':200,'runs':rows,'all_runs_valid':all(r['valid_long_output'] for r in rows)}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8'); print('RESULT '+json.dumps(result),flush=True); _=dll_dir
if __name__=='__main__': main()




