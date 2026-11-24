import argparse,json
from pathlib import Path
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
rows=[]
for line in a.input.read_text(encoding='utf-8-sig').splitlines():
 r=json.loads(line); msgs=r['messages']; prefix=tok.apply_chat_template(msgs[:-1],tokenize=True,add_generation_prompt=True); full=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=False)
 prefix=prefix['input_ids'];full=full['input_ids']
 if full[:len(prefix)]!=prefix: raise ValueError('chat-template prefix mismatch; refusing unsafe label mask')
 ids=list(full); labels=[-100]*len(ids)
 # Supervise assistant body only; template suffix EOT/newline remain masked.
 body=tok(msgs[-1]['content'],add_special_tokens=False)['input_ids']; start=len(prefix)
 if ids[start:start+len(body)]!=body: raise ValueError('assistant body tokenization mismatch')
 labels[start:start+len(body)]=body
 rows.append({'id':r['id'],'sourceId':r['sourceId'],'input_ids':ids,'labels':labels})
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')
print(json.dumps({'rows':len(rows),'supervised_tokens':sum(x!=-100 for r in rows for x in r['labels']),'output':str(a.output)}))


