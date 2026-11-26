"""Experimental Gemma mobile-QAT LoRA trainer with fixed-scale fake quantization."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.integrations.gemma_quant import QuantizedLinear, apply_srq
from transformers import AutoModelForImageTextToText

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model_loading import install_per_layer_scale_loader_patch  # noqa: E402
from quantization import pack_signed, unpack_signed  # noqa: E402
from training_diagnostics import gradient_diagnostics  # noqa: E402

BASE_DIR = Path(r"D:\Forge-GM\experiment\mobile_qat_feasibility\cache")
BASE_REVISION = "8e3ac78f1a4a97c19f5f194ab362a2206f71d961"
BITS_TO_DTYPE = {2: torch.uint8, 4: torch.uint8, 8: torch.int8}


def _srq_ste(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Exact source SRQ forward with identity STE; source apply_srq alone blocks all gradients."""
    quantized = apply_srq(x, scale)
    return x + (quantized - x).detach()


def _fixed_quant(weight: torch.Tensor, scale: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return straight-through effective weights, signed integers and pre-round normalized values."""
    if bits not in (2, 4, 8):
        raise ValueError(bits)
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    scale = scale.to(device=weight.device, dtype=torch.float32)
    normalized = weight.float() / scale
    clipped = normalized.clamp(low, high)
    integers = torch.round(clipped).to(torch.int8)
    rounded = integers.float() * scale
    # STE through rounding; clamp blocks gradients for values outside the deployed range.
    effective = (clipped * scale) + (rounded - clipped * scale).detach()
    return effective, integers, normalized


class QATLoRALinear(nn.Module):
    """QuantizedLinear with a merged low-rank delta quantized at original fixed scales."""
    def __init__(self, base: QuantizedLinear, rank: int = 16, alpha: float = 32.0,
                 dropout: float = 0.05):
        super().__init__()
        self.base_layer = base
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = dropout
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device,
                                                dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                                dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self._last_quant_stats: dict[str, int] = {}
        self._forward_calls = 0
        self._cached_eval_weight: torch.Tensor | None = None

    def train(self, mode: bool = True):
        if mode:
            self._cached_eval_weight = None
        return super().train(mode)

    @property
    def bits(self) -> int:
        return int(self.base_layer.num_bits)

    def delta_weight(self, stochastic: bool | None = None) -> torch.Tensor:
        # Elementwise dropout on A preserves the fixed-quantized-weight forward path.
        # It is explicitly recorded as factor-element dropout (p=0.05), not PEFT input dropout.
        use_dropout = self.training if stochastic is None else stochastic
        a = F.dropout(self.lora_A, p=self.dropout, training=use_dropout)
        return (self.lora_B @ a) * self.scaling

    def quantized_state(self, stochastic: bool | None = None):
        base = self.base_layer._dequantize_weights(dtype=torch.float32).detach()
        candidate = base + self.delta_weight(stochastic=stochastic)
        effective, integers, normalized = _fixed_quant(candidate, self.base_layer.weight_scale, self.bits)
        base_int = unpack_signed(self.base_layer.weight.detach(), self.base_layer.in_features, self.bits)
        sat = int(((normalized < -(1 << (self.bits - 1))) |
                   (normalized > (1 << (self.bits - 1)) - 1)).sum().item())
        changed = int((integers != base_int).sum().item())
        return effective, integers, normalized, changed, sat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._forward_calls += 1
        x = _srq_ste(x, self.base_layer.input_activation_scale)
        if self.training:
            weight, _, _, changed, saturated = self.quantized_state()
            self._last_quant_stats = {"changed_values": changed, "saturated_values": saturated}
        elif self._cached_eval_weight is None:
            weight = self.quantized_state(stochastic=False)[0].detach()
            self._cached_eval_weight = weight
        else:
            weight = self._cached_eval_weight
        out = F.linear(x, weight.to(dtype=x.dtype), self.base_layer.bias)
        return _srq_ste(out, self.base_layer.output_activation_scale)


def _replace_targets(model: nn.Module) -> dict[str, QATLoRALinear]:
    targets: dict[str, QATLoRALinear] = {}
    for name, module in list(model.named_modules()):
        if not name.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj")):
            continue
        if ".language_model.layers." not in name or ".self_attn." not in name:
            continue
        if not isinstance(module, QuantizedLinear):
            raise TypeError(f"target is not original packed QuantizedLinear: {name}: {type(module)}")
        wrapped = QATLoRALinear(module)
        model.set_submodule(name, wrapped)
        targets[name] = wrapped
    projection_counts = Counter(name.rsplit(".", 1)[-1] for name in targets)
    # Transformers intentionally omits duplicate K/V projections in Gemma shared-KV layers;
    # the corresponding packed source tensors remain frozen and are copied unchanged at merge.
    expected = {"q_proj": 42, "k_proj": 24, "v_proj": 24, "o_proj": 42}
    if dict(projection_counts) != expected:
        raise RuntimeError(f"mapped target inventory differs from shared-KV-aware expectation: {dict(projection_counts)}")
    if any(target.bits != 4 for target in targets.values()):
        raise RuntimeError("target attention projections are not all original INT4")
    return targets


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def quantized_inventory(targets: dict[str, QATLoRALinear], *, stochastic: bool = False) -> dict[str, Any]:
    by_bits: Counter[int] = Counter()
    changed_by_bits: Counter[int] = Counter()
    saturated_by_bits: Counter[int] = Counter()
    pack_mismatches = 0
    modules: list[dict[str, Any]] = []
    total_values = 0
    all_hash = hashlib.sha256()
    for name, layer in sorted(targets.items()):
        _, integers, normalized, changed, saturated = layer.quantized_state(stochastic=stochastic)
        packed = pack_signed(integers, layer.bits)
        # Exact round-trip is also the invariant used by the fixed-scale merge path.
        roundtrip = unpack_signed(packed, layer.base_layer.in_features, layer.bits)
        pack_mismatches += int(torch.count_nonzero(roundtrip != integers).item())
        packed_bytes = packed.detach().cpu().contiguous().numpy().tobytes()
        module_hash = hashlib.sha256(packed_bytes).hexdigest()
        all_hash.update(name.encode("utf-8") + b"\0" + packed_bytes)
        values = integers.numel()
        by_bits[layer.bits] += 1
        changed_by_bits[layer.bits] += changed
        saturated_by_bits[layer.bits] += saturated
        total_values += values
        modules.append({"module": name, "bits": layer.bits, "logical_values": values,
                        "changed_values": changed, "saturated_values": saturated,
                        "packed_sha256": module_hash,
                        "delta_fro_norm": float(layer.delta_weight(stochastic=False).norm().item())})
    return {"module_tensors_by_bits": {str(k): v for k, v in sorted(by_bits.items())},
            "changed_values_by_bits": {str(k): v for k, v in sorted(changed_by_bits.items())},
            "saturated_values_by_bits": {str(k): v for k, v in sorted(saturated_by_bits.items())},
            "logical_target_values": total_values, "pack_roundtrip_mismatches": pack_mismatches,
            "packed_fake_quant_sha256": all_hash.hexdigest(),
            "modules": modules}


def _load_model(model_dir: Path, estimate_mib: int, gate_log: Path) -> tuple[nn.Module, dict[str, QATLoRALinear]]:
    from gpu_gate import wait_for_gpu
    wait_for_gpu(estimate_mib, margin_mib=2048, consecutive=1, poll_seconds=30, log_path=gate_log)
    install_per_layer_scale_loader_patch()
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0},
        low_cpu_mem_usage=True, max_memory={0: "22GiB"},
    )
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    targets = _replace_targets(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, targets


def validate_layers(output: Path) -> dict[str, Any]:
    """Preflight on real Gemma packed linears, then test gradients and optimizer signal."""
    from transformers.integrations.gemma_quant import QuantizedLinear
    torch.manual_seed(451)
    base = QuantizedLinear(8, 4, bias=False, num_bits=4)
    signed = torch.tensor([[-8, -7, -2, -1, 0, 1, 6, 7], [1, 2, 3, 4, 5, 6, 7, -8],
                           [0, 1, -1, 2, -2, 3, -3, 4], [7, 6, 5, 4, 3, 2, 1, 0]], dtype=torch.int8)
    base.weight.data.copy_(pack_signed(signed, 4))
    base.weight_scale.data.fill_(0.25)
    base.input_activation_scale.data.fill_(0.25); base.output_activation_scale.data.fill_(0.5)
    wrapper = QATLoRALinear(base, rank=2, alpha=4, dropout=0.0)
    x = torch.randn(3, 8)
    before_weight = base.weight.detach().clone(); before_scale = base.weight_scale.detach().clone()
    with torch.no_grad():
        expected = base(x)
        actual = wrapper(x)
    if not torch.equal(expected, actual):
        raise AssertionError("zero-delta QAT forward differs from original quantized base")
    # A controlled delta crosses one known fixed-scale quantization threshold.
    with torch.no_grad():
        wrapper.lora_A.zero_(); wrapper.lora_B.zero_()
        wrapper.lora_A[0, 0] = 1.0; wrapper.lora_B[0, 0] = 0.16
        _, ints, _, changed, _ = wrapper.quantized_state(stochastic=False)
    if changed < 1 or int(ints[0, 0]) != -7:
        raise AssertionError("controlled LoRA delta failed predictable fixed-scale quantization")
    # Reset to a nonzero factorization so both A and B receive gradients through STE.
    with torch.no_grad():
        wrapper.lora_A.normal_(0, 0.1); wrapper.lora_B.normal_(0, 0.1)
    target = torch.randn(3, 4)
    optimizer = torch.optim.AdamW([wrapper.lora_A, wrapper.lora_B], lr=1e-2, weight_decay=0.0,
                                  foreach=False)
    losses=[]
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        loss=F.mse_loss(wrapper(x),target); loss.backward()
        if wrapper.lora_A.grad is None or wrapper.lora_B.grad is None:
            raise AssertionError("LoRA gradients missing through fake quantizer")
        if wrapper.lora_A.grad.norm().item()==0 or wrapper.lora_B.grad.norm().item()==0:
            raise AssertionError("LoRA gradient is zero through fake quantizer")
        torch.nn.utils.clip_grad_norm_([wrapper.lora_A,wrapper.lora_B],1.0)
        optimizer.step(); losses.append(float(loss.item()))
    if losses[-1] >= losses[0]:
        raise AssertionError(f"short optimization did not reduce loss: {losses}")
    packed=pack_signed(unpack_signed(base.weight,base.in_features,4),4)
    if not torch.equal(packed,before_weight) or not torch.equal(base.weight_scale,before_scale):
        raise AssertionError("frozen base tensor or scale changed")
    record={"zero_delta_base_output_exact":True,"controlled_delta_crossed_quantizer":True,
            "changed_values_after_control":changed,"gradients_reached_A_and_B":True,
            "base_and_scales_unchanged":True,"pack_depack_exact":True,
            "optimizer_loss_first":losses[0],"optimizer_loss_last":losses[-1],
            "optimizer_loss_decreased":True,"qat_ste":"fixed-scale weight round/clamp STE plus exact SRQ forward with identity activation STE",
            "real_attention_targets":"verified separately during full-model load"}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    return record


def _adapter_state(targets: dict[str, QATLoRALinear]) -> dict[str, torch.Tensor]:
    out={}
    for name, layer in targets.items():
        out[f"{name}.lora_A.weight"]=layer.lora_A.detach().cpu().contiguous()
        out[f"{name}.lora_B.weight"]=layer.lora_B.detach().cpu().contiguous()
    return out


def _save_checkpoint(path: Path, targets: dict[str, QATLoRALinear], optimizer, global_step: int,
                     sample_index: int, loss_sum: float, rng_seed: int, arm: str) -> None:
    from safetensors.torch import save_file
    path.mkdir(parents=True,exist_ok=False)
    save_file(_adapter_state(targets),str(path/'adapter_model.safetensors'),metadata={"format":"pt","status":"EXPERIMENTAL_NOT_GOLD"})
    torch.save({"optimizer":optimizer.state_dict(),"global_step":global_step,"sample_index":sample_index,
                "loss_sum":loss_sum,"python_rng":random.getstate(),"torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state_all(),"seed":rng_seed,"arm":arm,
                "base_revision":BASE_REVISION},path/'training_state.pt')
    (path/'adapter_config.json').write_text(json.dumps({"status":"EXPERIMENTAL_NOT_GOLD","rank":16,"alpha":32,
        "factor_dropout":0.05,"dropout_semantics":"elementwise dropout over LoRA A factor entries to retain fixed-scale fake-quant forward",
        "target_modules":["q_proj","k_proj","v_proj","o_proj"],"base_revision":BASE_REVISION},indent=2)+'\n',encoding='utf-8')


def train(arm: str, prepared: Path, out: Path, *, max_samples: int | None = None,
          estimate_mib: int = 20480, resume: Path | None = None, model_dir: Path = BASE_DIR) -> dict[str, Any]:
    seed=451
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rows=_load_jsonl(prepared)
    if max_samples is not None: rows=rows[:max_samples]
    if not rows: raise ValueError("empty training set")
    permutation=list(range(len(rows))); random.Random(seed).shuffle(permutation)
    rows=[rows[i] for i in permutation]
    model,targets=_load_model(model_dir,estimate_mib,out/'gpu_gate.jsonl')
    trainable=[p for layer in targets.values() for p in (layer.lora_A,layer.lora_B)]
    optimizer=torch.optim.AdamW(trainable,lr=1e-4,weight_decay=0.01,foreach=False,fused=True)
    optimizer.zero_grad(set_to_none=True)
    steps=math.ceil(len(rows)/8); warmup=max(1,round(steps*0.05)); global_step=0
    def lr_for(step:int)->float:
        if step<warmup:return (step+1)/warmup
        return 0.5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,steps-warmup))))
    def apply_lr():
        factor=lr_for(global_step)
        for group in optimizer.param_groups:group['lr']=1e-4*factor
        return 1e-4*factor
    start_idx=0; losses=[]
    if resume:
        state=torch.load(resume/'training_state.pt',map_location='cpu',weights_only=False)
        from safetensors.torch import load_file
        adapter=load_file(str(resume/'adapter_model.safetensors'),device='cpu')
        for name,layer in targets.items():
            layer.lora_A.data.copy_(adapter[f'{name}.lora_A.weight'].to(layer.lora_A.device))
            layer.lora_B.data.copy_(adapter[f'{name}.lora_B.weight'].to(layer.lora_B.device))
        optimizer.load_state_dict(state['optimizer']); global_step=state['global_step']; start_idx=state['sample_index']
        random.setstate(state['python_rng']); torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    started=time.monotonic(); metrics=out/'metrics.jsonl'; out.mkdir(parents=True,exist_ok=True)
    # Keep intermediate marks on optimizer boundaries, but save 100% only at
    # the true end of the split so a short final partial batch is included.
    fraction_marks={max(1,int(round(len(rows)*f/8))*8):f for f in (0.25,0.5,0.75)}
    fraction_marks[len(rows)]=1.0
    cumulative_supervised=[]; running_supervised=0
    for item in rows:
        running_supervised += sum(v != -100 for v in item['labels'])
        cumulative_supervised.append(running_supervised)
    accumulated=0.0; seen=0; families=set(); optimizer.zero_grad(set_to_none=True)
    model.train()
    for index in range(start_idx,len(rows)):
        row=rows[index]; ids=torch.tensor(row['input_ids'],device='cuda',dtype=torch.long).unsqueeze(0)
        labels=torch.tensor(row['labels'],device='cuda',dtype=torch.long).unsqueeze(0)
        attention=torch.ones_like(ids)
        group_start=(index//8)*8; group_size=min(8,len(rows)-group_start)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=model(input_ids=ids,attention_mask=attention,labels=labels,use_cache=False)
            loss=result.loss
        # One scalar transfer instead of separate finite-check and loss transfers.
        loss_value=float(loss.detach().item())
        if not math.isfinite(loss_value):raise FloatingPointError(f'nonfinite loss at sample {index}')
        (loss/group_size).backward(); accumulated+=loss_value;seen+=1
        loss_requires_grad=bool(loss.requires_grad)
        logits_requires_grad=bool(result.logits.requires_grad)
        # Do not keep the previous example's potentially enormous logits alive
        # during the next forward pass.
        del loss,result
        source=row.get('sourceId','unknown');families.add(str(source).split(':')[0])
        is_group_end=(index+1)%8==0 or index+1==len(rows)
        if is_group_end:
            grad_tensors_present,grad_tensors_nonzero,nonzero_grad_norm_sum,grad_norm=gradient_diagnostics(trainable,1.0)
            lr=apply_lr();optimizer.step();optimizer.zero_grad(set_to_none=True);global_step+=1
            torch.cuda.synchronize()
            mem_free,mem_total=torch.cuda.mem_get_info()
            rec={'time_utc':datetime.now(timezone.utc).isoformat(),'arm':arm,'global_step':global_step,
                 'samples_processed':index+1,'samples_in_update':group_size,'effective_batch':group_size,
                 'supervised_tokens_processed':cumulative_supervised[index],
                 'mean_loss_update':accumulated/group_size,'lr':lr,'grad_norm':grad_norm,
                 'families_seen':len(families),'vram_free_mib':mem_free//(1024**2),
                 'vram_total_mib':mem_total//(1024**2),'vram_peak_allocated_mib':torch.cuda.max_memory_allocated()//(1024**2),
                 'grad_tensors_present':grad_tensors_present,
                 'grad_tensors_nonzero':grad_tensors_nonzero,
                 'target_modules_called':sum(layer._forward_calls>0 for layer in targets.values()),
                 'loss_requires_grad':loss_requires_grad,
                 'logits_requires_grad':logits_requires_grad,
                 'nonzero_grad_norm_sum':nonzero_grad_norm_sum,
                 'host_ram_bytes':__import__('psutil').virtual_memory().used,'elapsed_seconds':round(time.monotonic()-started,2)}
            _write_jsonl(metrics,rec);losses.append(accumulated/group_size);accumulated=0.0
            print(json.dumps(rec),flush=True)
        if (index+1) in fraction_marks:
            fraction=fraction_marks[index+1]
            model.eval();qstats=quantized_inventory(targets,stochastic=False);model.train()
            ckpt=out/'checkpoints'/f'epoch_{int(fraction*100):03d}'
            if fraction == 1.0 and ckpt.exists():
                ckpt=out/'checkpoints'/'epoch_100_final'
            _save_checkpoint(ckpt,targets,optimizer,global_step,index+1,accumulated,seed,arm)
            _write_jsonl(out/'quantization_metrics.jsonl',{'arm':arm,'fraction':fraction,'samples_processed':index+1,
                'global_step':global_step,'quantized':{k:v for k,v in qstats.items() if k!='modules'},
                'per_module':qstats['modules'],'checkpoint_adapter_sha256':_sha256(ckpt/'adapter_model.safetensors')})
            print(json.dumps({'checkpoint':str(ckpt),'fraction':fraction,'changed_values_by_bits':qstats['changed_values_by_bits'],
                              'saturated_values_by_bits':qstats['saturated_values_by_bits'],'pack_roundtrip_mismatches':qstats['pack_roundtrip_mismatches']}),flush=True)
    final={'arm':arm,'status':'EXPERIMENTAL_NOT_GOLD','base_revision':BASE_REVISION,'samples':len(rows),
           'global_steps':global_step,'loss_first_update':losses[0] if losses else None,
           'loss_last_update':losses[-1] if losses else None,'loss_reduced':bool(losses and losses[-1]<losses[0]),
           'elapsed_seconds':round(time.monotonic()-started,2),'optimizer':'torch.optim.AdamW(fused=True,foreach=False)',
           'rank':16,'alpha':32,'factor_dropout':0.05,'target_module_count':len(targets),
           'effective_batch':8,'last_group_effective_batch':len(rows)%8 or 8,'sequence_limit':4096,
           'sequence_truncation':False,'base_init_only':True}
    (out/'arm_summary.json').write_text(json.dumps(final,indent=2)+'\n',encoding='utf-8')
    del model,targets,optimizer;torch.cuda.empty_cache()
    return final


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--validate-layers',type=Path)
    parser.add_argument('--arm',choices=('P0','P10','P20'))
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-samples',type=int)
    parser.add_argument('--estimate-gib',type=float,default=20.0)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--model-dir',type=Path,default=BASE_DIR)
    args=parser.parse_args()
    if args.validate_layers:
        print(json.dumps(validate_layers(args.validate_layers),indent=2));return
    if not all((args.arm,args.prepared,args.output)):parser.error('--arm, --prepared and --output are required for training')
    print(json.dumps(train(args.arm,args.prepared,args.output,max_samples=args.max_samples,
                           estimate_mib=round(args.estimate_gib*1024),resume=args.resume,model_dir=args.model_dir),indent=2))

if __name__=='__main__':main()


