"""No model downloads required. Run: python -m unittest discover -s repro/tests -v"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from training_diagnostics import gradient_diagnostics


class GradientDiagnosticsTests(unittest.TestCase):
    def compare_legacy(self, device, raw, max_norm=1.0):
        def make():
            params = []
            for value in raw:
                p = torch.nn.Parameter(torch.zeros(3, device=device))
                if value is not None:
                    p.grad = torch.tensor(value, device=device, dtype=p.dtype)
                params.append(p)
            return params

        expected, actual = make(), make()
        grads = [p.grad for p in expected if p.grad is not None]
        old = (len(grads), sum(bool(torch.count_nonzero(g).item()) for g in grads),
               float(sum(g.float().norm().item() for g in grads)),
               float(torch.nn.utils.clip_grad_norm_(expected, max_norm).item()))
        got = gradient_diagnostics(actual, max_norm)
        self.assertEqual(got, old)
        for p, q in zip(expected, actual):
            if p.grad is None:
                self.assertIsNone(q.grad)
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    def test_cpu(self):
        for raw in ([[3., 4., 0.], None, [0., 0., 0.], [-2., 1., 0.]],
                    [None, None], [], [[0., 0., 0.]], [[0.01, 0., 0.]]):
            with self.subTest(raw=raw):
                self.compare_legacy("cpu", raw)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda(self):
        self.compare_legacy("cuda", [[3., 4., 0.], None, [0., 0., 0.], [-2., 1., 0.]])

    def test_optimizer_step_unchanged(self):
        a = torch.nn.Parameter(torch.tensor([1., 2., 3.]))
        b = torch.nn.Parameter(a.detach().clone())
        oa = torch.optim.AdamW([a], lr=1e-4)
        ob = torch.optim.AdamW([b], lr=1e-4)
        for _ in range(3):
            oa.zero_grad(set_to_none=True)
            ob.zero_grad(set_to_none=True)
            (a.square().sum()).backward()
            (b.square().sum()).backward()
            torch.nn.utils.clip_grad_norm_([a], 1.)
            gradient_diagnostics([b], 1.)
            oa.step()
            ob.step()
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
