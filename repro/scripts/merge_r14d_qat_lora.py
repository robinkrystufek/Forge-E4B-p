"""Merge a QAT-LoRA adapter into pinned Gemma mobile-QAT weights at fixed scales."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from collections import Counter
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from quantization import dequantize, pack_signed, quantize_fixed_scales, unpack_signed
from round_trip_checkpoint import scale_key


def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def merge(source:Path,manifest_path:Path,adapter_dir:Path,output:Path,expected_metrics:Path|None=None)->dict:
 if output.exists():raise FileExistsError(f'refusing existing merged checkpoint: {output}')
 manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
 headers=manifest['safetensors']['tensor_headers']
 from safetensors.torch import load_file
 adapter=load_file(str(adapter_dir/'adapter_model.safetensors'),device='cpu')
 module_names=sorted({key.rsplit('.lora_',1)[0] for key in adapter})
 expected_keys={f'{n}.lora_{factor}.weight' for n in module_names for factor in ('A','B')}
 if set(adapter)!=expected_keys:raise ValueError('adapter A/B key inventory is incomplete or unexpected')
 out_tensors={};changed_by_bits=Counter();changed_tensors_by_bits=Counter();total_by_bits=Counter();packed_digest=hashlib.sha256();module_rows=[]
 with safe_open(str(source/'model.safetensors'),framework='pt',device='cpu') as f:
  metadata=f.metadata()
  for key in f.keys():out_tensors[key]=f.get_tensor(key)
  for module in module_names:
   weight_key=f'{module}.weight'; scale_name=scale_key(weight_key)
   if weight_key not in headers or headers[weight_key].get('assigned_bits') not in (2,4,8):
    raise ValueError(f'adapter target is not a packed source weight: {weight_key}')
   spec=headers[weight_key];bits=int(spec['assigned_bits']);packed=f.get_tensor(weight_key);scale=f.get_tensor(scale_name)
   width=spec['shape'][-1]*(8//bits) if bits<8 else spec['shape'][-1]
   a=adapter[f'{module}.lora_A.weight'].float();b=adapter[f'{module}.lora_B.weight'].float()
   if a.shape[1]!=width or b.shape[0]!=spec['shape'][0] or a.shape[0]!=16 or b.shape[1]!=16:
    raise ValueError(f'LoRA matrix shape mismatch for {module}: A{tuple(a.shape)} B{tuple(b.shape)} source{spec["shape"]}')
   delta=(b@a)*(32.0/16.0)
   effective=dequantize(packed,scale,width,bits,dtype=torch.float32)+delta
   updated=quantize_fixed_scales(effective,scale,bits)
   ints=unpack_signed(updated,width,bits)
   base_ints=unpack_signed(packed,width,bits)
   values=ints.numel();diff=int(torch.count_nonzero(ints!=base_ints).item())
   packed_bytes=updated.contiguous().numpy().tobytes();digest=hashlib.sha256(packed_bytes).hexdigest()
   packed_digest.update(module.encode('utf-8')+b'\0'+packed_bytes)
   out_tensors[weight_key]=updated.contiguous()
   changed_by_bits[bits]+=diff;changed_tensors_by_bits[bits]+=int(diff>0);total_by_bits[bits]+=values
   module_rows.append({'module':module,'tensor':weight_key,'bits':bits,'logical_values':values,'changed_values':diff,
                       'packed_sha256':digest,'scale_sha256':hashlib.sha256(scale.contiguous().numpy().tobytes()).hexdigest(),
                       'delta_fro_norm':float(delta.norm().item())})
 output.mkdir(parents=True)
 tmp=output/'model.safetensors.tmp'
 save_file(out_tensors,str(tmp),metadata=metadata)
 tmp.replace(output/'model.safetensors')
 for path in source.iterdir():
  if path.is_file() and path.name!='model.safetensors':shutil.copy2(path,output/path.name)
 expected_hash=None
 if expected_metrics:
  entries=[json.loads(x) for x in expected_metrics.read_text(encoding='utf-8').splitlines() if x]
  expected_hash=entries[-1]['quantized']['packed_fake_quant_sha256']
 merged_hash=packed_digest.hexdigest()
 result={'status':'EXPERIMENTAL_NOT_GOLD','source_revision':manifest['revision'],'source_model_safetensors_sha256':sha256(source/'model.safetensors'),
         'adapter':str(adapter_dir),'adapter_sha256':sha256(adapter_dir/'adapter_model.safetensors'),
         'output_dir':str(output),'merged_model_safetensors_sha256':sha256(output/'model.safetensors'),
         'target_module_count':len(module_names),'changed_tensors_by_bits':{str(k):v for k,v in sorted(changed_tensors_by_bits.items())},
         'changed_values_by_bits':{str(k):v for k,v in sorted(changed_by_bits.items())},
         'total_target_values_by_bits':{str(k):v for k,v in sorted(total_by_bits.items())},
         'training_fake_quant_packed_sha256':expected_hash,'merged_packed_sha256':merged_hash,
         'training_merge_packed_exact':expected_hash==merged_hash if expected_hash else None,
         'method':'source packed -> float32 dequant -> merged LoRA -> fixed original per-row scale -> original Gemma offset-binary packing',
         'modules':module_rows}
 report=output/'fixed_scale_merge_report.json';report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 if expected_hash and expected_hash!=merged_hash:
  raise RuntimeError(f'training fake-quant packed hash mismatch: {expected_hash} != {merged_hash}; preserve output for diagnosis')
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-model',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
 p.add_argument('--adapter-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-metrics',type=Path)
 a=p.parse_args();print(json.dumps(merge(a.source_model,a.manifest,a.adapter_dir,a.output,a.expected_metrics),indent=2))
if __name__=='__main__':main()
