"""Update-level gradient diagnostics without per-parameter host synchronization."""
from __future__ import annotations

import torch


@torch.no_grad()
def gradient_diagnostics(parameters, max_norm: float) -> tuple[int, int, float, float]:
    """Return legacy preclip metrics and clip once, with one host transfer per device.

    Keep the original FP32 per-tensor norms and Python summation order.
    This trainer uses dense, single-device gradients. Grouping by device also
    supports CPU tests and avoids stacking tensors from different devices.
    No change to clipping, optimizer, gradient accumulation or loss weighting.
    """
    parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    by_device = {}
    for index, grad in enumerate(grads):
        # Capture preclip scalars before clip_grad_norm_ mutates gradients.
        nonzero = torch.count_nonzero(grad).ne(0).to(torch.float32)
        norm = grad.float().norm()
        by_device.setdefault(grad.device, []).append((index, nonzero, norm))

    clipped_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    by_device.setdefault(clipped_norm.device, [])
    values = [0.0] * len(grads)
    nonzero_count = 0
    total_norm = 0.0
    for device, records in by_device.items():
        scalars = [value for _, nonzero, norm in records for value in (nonzero, norm)]
        if device == clipped_norm.device:
            scalars.append(clipped_norm)
        # A single transfer for all scalar diagnostics on this device.
        host = torch.stack(scalars).detach().cpu().tolist()
        for pos, (index, _, _) in enumerate(records):
            nonzero_count += int(host[2 * pos])
            values[index] = float(host[2 * pos + 1])
        if device == clipped_norm.device:
            total_norm = float(host[-1])
    return len(grads), nonzero_count, float(sum(values)), total_norm
