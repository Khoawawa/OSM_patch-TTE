import time
import torch
from collections import OrderedDict

class StageTimer:
    """
    Timer for measuring time BETWEEN model components.
    Designed for percentage runtime attribution.
    """
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self._start = {}
        self.records = OrderedDict()

    def _sync(self):
        if self.use_cuda:
            torch.cuda.synchronize()

    def begin(self, name):
        self._sync()
        self._start[name] = time.perf_counter()

    def end(self, name):
        self._sync()
        elapsed = time.perf_counter() - self._start.pop(name)
        self.records[name] = self.records.get(name, 0.0) + elapsed

    def summary(self):
        total = sum(self.records.values())
        print("\n=== Forward Runtime Breakdown ===")
        for k, t in self.records.items():
            pct = 100 * t / total if total > 0 else 0.0
            print(f"{k:25s} {t:.6f}s  ({pct:5.1f}%)")
        print(f"{'TOTAL':25s} {total:.6f}s  (100.0%)")