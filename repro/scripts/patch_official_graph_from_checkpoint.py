"""Patch verified fixed-scale Gemma checkpoint payloads into the official TFLite graph."""
from __future__ import annotations
import argparse,hashlib,json,mmap,re,shutil
from pathlib import Path
import numpy as np
from safetensors import safe_open
from quantization import unpack_signed
from round_trip_checkpoint import scale_key
from patch_official_qat_graph import _buffer_ranges,_decode_tflite,_graph_schema,_sha256

def _file_sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def _pack_tflite(values:np.ndarray,bits:int)->bytes:
 values=np.asarray(values,dtype=np.int8)
 if bits==8:return values.tobytes()
 if bits not in (2,4):raise ValueError(bits)
 factor=8//bits;mask=(1<<bits)-1
 flat=values.reshape(-1).astype(np.int16)
 padded=np.pad(flat,(0,(-flat.size)%factor))
 groups=padded.reshape(-1,factor)
 result=np.zeros(groups.shape[0],dtype=np.uint8)
 for i in range(factor):result |= ((groups[:,i]&mask).astype(np.uint8) << (i*bits))
 return result.tobytes()

def patch(graph:Path,provenance:Path,merged:Path,merge_report:Path,source_manifest:Path,
          output:Path,manifest_path:Path)->dict:
 if output.exists() or manifest_path.exists():raise FileExistsError('refusing to overwrite graph or patch manifest')
 audit=json.loads(provenance.read_text(encoding='utf-8'))
 merge=json.loads(merge_report.read_text(encoding='utf-8'))
 source_meta=json.loads(source_manifest.read_text(encoding='utf-8'))
 if merge.get('training_merge_packed_exact') is not True:
  raise ValueError('refusing to package: fixed-scale merged payload does not exactly match training fake-quant payload')
 matches={row['source_tensor']:row for row in audit['source_tensor_matches_by_graph']['official_main']}
 changed=[row for row in merge['modules'] if row['changed_values']>0]
 if not changed:raise ValueError('selected adapter has no fixed-scale packed weight changes')
 plan=[];buffer_ids=set()
 with safe_open(str(merged/'model.safetensors'),framework='pt',device='cpu') as f:
  for mod in changed:
   name=mod['tensor'];row=matches.get(name)
   if row is None:raise ValueError(f'no source-to-official audit row for changed tensor: {name}')
   base=[m for m in row.get('matches',[]) if m.get('revision_match')=='base']
   if len(base)!=1 or not base[0].get('scale_exact'):raise ValueError(f'no unique exact base mapping for {name}')
   match=base[0];graph_id=match['graph_tensor_id']
   bm=re.search(r':buffer(\d+):INT([248]):([0-9x]+):',graph_id)
   if not bm:raise ValueError(f'unrecognized official graph tensor id {graph_id}')
   bid=int(bm.group(1));bits=int(bm.group(2));shape=tuple(int(x) for x in bm.group(3).split('x'))
   if bits!=int(row['source_bits']) or list(shape)!=row['logical_shape'] or bits!=int(mod['bits']):
    raise ValueError(f'official/source tensor width or shape mismatch for {name}')
   wk=name;sk=scale_key(wk);headers=source_meta['safetensors']['tensor_headers']
   if headers[wk]['assigned_bits']!=bits:raise ValueError(f'source manifest width mismatch for {name}')
   scale=f.get_tensor(sk)
   scale_hash=_sha256(np.asarray(scale.cpu(),dtype=np.float32))
   if scale_hash!=match['graph_scales_sha256']:raise ValueError(f'official scale mismatch for {name}')
   packed=f.get_tensor(wk)
   width=shape[-1];ints=unpack_signed(packed,width,bits).cpu().numpy().astype(np.int8,copy=False)
   current_integer_hash=hashlib.sha256(np.ascontiguousarray(ints).tobytes()).hexdigest()
   payload=_pack_tflite(ints,bits)
   plan.append({'source_tensor':name,'bits':bits,'logical_shape':list(shape),'official_buffer':bid,
                'scale_sha256':scale_hash,'base_integer_values_sha256':row['base_integer_values_sha256'],
                'modified_integer_values_sha256':current_integer_hash,
                'payload_sha256':hashlib.sha256(payload).hexdigest(),'changed_values':mod['changed_values'],
                'payload':payload})
   buffer_ids.add(bid)
 meta,_=_graph_schema(graph,buffer_ids);ranges=_buffer_ranges(graph,buffer_ids)
 for item in plan:
  info=meta[item['official_buffer']];start,size=ranges[item['official_buffer']]
  if size!=len(item['payload']):raise ValueError(f'packed range size mismatch {item["source_tensor"]}: {size} != {len(item["payload"])}')
  if (info['bits'],info['shape'],info['scale_sha256'],info['quantized_dimension'])!=(item['bits'],tuple(item['logical_shape']),item['scale_sha256'],0):
   raise ValueError(f'graph tensor metadata mismatch: {item["source_tensor"]}: {info}')
  item['offset']=start;item['length']=size
 # Validate every official payload still represents the pinned base before changing it.
 with graph.open('rb') as f:
  mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
  for item in plan:
   start,size=ranges[item['official_buffer']]
   actual=_sha256(_decode_tflite(mm[start:start+size],item['bits'],int(np.prod(item['logical_shape']))))
   if actual!=item['base_integer_values_sha256']:raise ValueError(f'official graph no longer matches pinned base for {item["source_tensor"]}')
  mm.close()
 output.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(graph,output)
 with output.open('r+b') as f:
  mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_WRITE)
  for item in plan:mm[item['offset']:item['offset']+item['length']]=item['payload']
  mm.flush();mm.close()
 with output.open('rb') as f:
  mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
  for item in plan:
   actual=_sha256(_decode_tflite(mm[item['offset']:item['offset']+item['length']],item['bits'],int(np.prod(item['logical_shape']))))
   if actual!=item['modified_integer_values_sha256']:raise ValueError(f'graph readback failed for {item["source_tensor"]}')
  mm.close()
 # Byte-compare all graph ranges outside the explicitly replaced payloads.
 ranges_sorted=sorted((x['offset'],x['length']) for x in plan)
 outside=hashlib.sha256()
 with graph.open('rb') as a,output.open('rb') as b:
  pos=0
  for start,size in ranges_sorted:
   a.seek(pos);b.seek(pos);remaining=start-pos
   while remaining:
    n=min(8<<20,remaining);aa=a.read(n);bb=b.read(n)
    if aa!=bb:raise ValueError('a non-payload graph byte changed')
    outside.update(aa);pos+=n;remaining-=n
   pos=start+size
  a.seek(pos);b.seek(pos)
  while True:
   aa=a.read(8<<20);bb=b.read(8<<20)
   if aa!=bb:raise ValueError('a non-payload graph byte changed')
   if not aa:break
   outside.update(aa)
 result={'status':'source_payloads_patched_into_official_graph','official_graph':str(graph),'official_graph_sha256':_file_sha(graph),
         'patched_graph':str(output),'patched_graph_sha256':_file_sha(output),'outside_payload_sha256':outside.hexdigest(),
         'outside_payload_bytes_unchanged':True,'patched_tensor_count':len(plan),
         'changed_tensors_by_bits':{str(b):sum(x['bits']==b for x in plan) for b in (2,4,8)},
         'changed_values_by_bits':{str(b):sum(x['changed_values'] for x in plan if x['bits']==b) for b in (2,4,8)},
         'source_revision':source_meta['revision'],'merge_packed_hash_match':merge['training_merge_packed_exact'],
         'patches':[{k:v for k,v in x.items() if k not in ('payload','offset','length')} for x in plan]}
 manifest_path.parent.mkdir(parents=True,exist_ok=True);manifest_path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--official-graph',type=Path,required=True);p.add_argument('--provenance',type=Path,required=True)
 p.add_argument('--merged-model',type=Path,required=True);p.add_argument('--merge-report',type=Path,required=True);p.add_argument('--source-manifest',type=Path,required=True)
 p.add_argument('--output-graph',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
 a=p.parse_args();print(json.dumps(patch(a.official_graph,a.provenance,a.merged_model,a.merge_report,a.source_manifest,a.output_graph,a.manifest),indent=2))
if __name__=='__main__':main()
