"""Rebuild an official LiteRT-LM bundle with only its main graph path replaced."""
from __future__ import annotations
import argparse,hashlib,io,json,re,shutil
from pathlib import Path
from litert_lm_builder import LitertLmFileBuilder,litertlm_peek,litertlm_header_schema_py_generated as schema

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

TEXT_ONLY_MODEL_TYPES = {
 'audio_encoder_hw', 'audio_adapter', 'end_of_audio',
 'vision_encoder', 'vision_adapter', 'end_of_vision',
}

def remove_multimodal_sections(text:str)->tuple[str,list[str]]:
 section_re=re.compile(r'(?ms)^\[\[section\]\]\s*\n.*?(?=^\[\[section\]\]|\Z)')
 blocks=section_re.findall(text);removed=[];kept=[]
 for block in blocks:
  match=re.search(r'(?m)^model_type\s*=\s*"([^"]+)"\s*$',block)
  model_type=match.group(1) if match else None
  if model_type in TEXT_ONLY_MODEL_TYPES:removed.append(model_type)
  else:kept.append(block)
 if set(removed)!=TEXT_ONLY_MODEL_TYPES:
  raise ValueError(f'expected all six audio/vision sections, found {removed}')
 first=section_re.search(text)
 if not first:raise ValueError('section table parse failed')
 return text[:first.start()]+''.join(kept),removed

def repack(unpacked:Path,patched_graph:Path,output:Path,manifest:Path,*,text_only:bool=False)->dict:
 if output.exists() or manifest.exists():raise FileExistsError('refusing to overwrite package or manifest')
 model_toml=unpacked/'model.toml';text=model_toml.read_text(encoding='utf-8')
 original_name='Section10_TFLiteModel_tf_lite_prefill_decode.tflite';seen=0
 def path_repl(match):
  nonlocal seen
  rel=match.group(1)
  if Path(rel).name==original_name:
   source=patched_graph;seen+=1
  else:source=unpacked/rel
  if not source.is_file():raise FileNotFoundError(source)
  return 'data_path = '+json.dumps(str(source.resolve()))
 updated=re.sub(r'data_path\s*=\s*"([^"]+)"',path_repl,text)
 if seen!=1:raise ValueError(f'expected exactly one official prefill/decode graph path, found {seen}')
 removed=[]
 if text_only:
  # Remove whole TOML sections so builder omits each graph and linked weights.
  updated,removed=remove_multimodal_sections(updated)
 manifest.parent.mkdir(parents=True,exist_ok=True);output.parent.mkdir(parents=True,exist_ok=True)
 toml_out=output.parent/'repack_model.toml';toml_out.write_text(updated,encoding='utf-8')
 staging=output.parent/'repack_staging';staging.mkdir(exist_ok=False)
 builder=LitertLmFileBuilder.from_toml_file(str(toml_out))
 with output.open('xb') as f:
  result=builder.build(f,validate_metadata=False,weights_in_model_graph=False,
                       externalization_work_dir=str(staging))
 header=litertlm_peek.read_litertlm_header(str(output),io.StringIO())
 sm=header.SectionMetadata();type_names={getattr(schema.AnySectionDataType,n):n for n in dir(schema.AnySectionDataType) if not n.startswith('_')}
 sections=[]
 for i in range(sm.ObjectsLength()):
  section=sm.Objects(i);sections.append({'type':type_names.get(section.DataType(),str(section.DataType())),
    'model_type':litertlm_peek.get_model_type(section),'bytes':int(section.EndOffset()-section.BeginOffset())})
 payload={'status':'official_graph_payload_patch_repacked','package':str(output),'bytes':output.stat().st_size,
  'sha256':sha256(output),'patched_graph':str(patched_graph),'patched_graph_sha256':sha256(patched_graph),
  'official_unpacked_dir':str(unpacked),'official_model_toml_sha256':sha256(model_toml),
  'model_toml_transform':'official definitions retained except six audio/vision section records removed; prefill graph data_path selected as requested',
  'section_count':len(sections),'sections':sections,'builder_result':getattr(result,'__dict__',str(result)),
  'text_only':text_only,'removed_multimodal_model_types':removed,
  'retained_required_model_types':['embedder','per_layer_embedder','prefill_decode','mtp_drafter'] if text_only else []}
 manifest.write_text(json.dumps(payload,indent=2,default=str)+'\n',encoding='utf-8')
 return payload

def main():
 p=argparse.ArgumentParser();p.add_argument('--unpacked-dir',type=Path,required=True);p.add_argument('--patched-graph',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
 p.add_argument('--text-only',action='store_true',help='Remove audio/vision model sections and their external weights; preserve shared text embedders and MTP')
 a=p.parse_args();print(json.dumps(repack(a.unpacked_dir,a.patched_graph,a.output,a.manifest,text_only=a.text_only),indent=2,default=str))
if __name__=='__main__':main()
