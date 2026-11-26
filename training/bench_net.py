"""Time one batched policy step (inference) for several batch sizes."""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))
from model import Net, sample

dev = sys.argv[1] if len(sys.argv) > 1 else ("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(dev).eval()
for B in (16, 32, 64):
    o = {"vec": torch.zeros(B, 66), "rays": torch.zeros(B, 7, 64), "ents": torch.zeros(B, 32, 6),
         "etype": torch.zeros(B, 32).long(), "map": torch.zeros(B, 8, 32, 32),
         "crop_idx": torch.zeros(B, 1024).long(), "pos_idx": torch.zeros(B).long()}
    o = {k: v.to(dev) for k, v in o.items()}
    st = net.init_state(B, dev)
    with torch.no_grad():
        f = lambda: sample(net.step(o, st)[0])[0].cpu()
        f(); t = time.perf_counter()
        for _ in range(30): f()
    print(dev, B, round((time.perf_counter() - t) / 30 * 1e3, 1), "ms")
