import argparse,hashlib,json,urllib.request
from pathlib import Path
REPO='litert-community/gemma-4-E4B-it-litert-lm'; REV='f254b5ba6e36f7d89cc7892e268e9877bc20f0a0'; FILE='gemma-4-E4B-it.litertlm'; SHA='0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0'
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix(a.output.suffix+'.partial');url=f'https://huggingface.co/{REPO}/resolve/{REV}/{FILE}?download=true'
with urllib.request.urlopen(url,timeout=120) as r,tmp.open('wb') as f:
 while b:=r.read(8*1024*1024):f.write(b)
h=hashlib.sha256()
with tmp.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
digest=h.hexdigest()
if digest!=SHA:tmp.unlink();raise SystemExit(f'stock package SHA mismatch: {digest}')
tmp.replace(a.output);print(json.dumps({'repo':REPO,'revision':REV,'file':FILE,'sha256':digest,'bytes':a.output.stat().st_size,'path':str(a.output)}))
