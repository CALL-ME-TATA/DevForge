"""WiSE-FT weight-space ensembling (Wortsman et al., CVPR 2022): theta = (1 - a) * old + a * finetuned.
Keeps the old checkpoint's skills while adding the fine-tuned robustness; no extra inference cost.

  python training/wise_ft.py old.pt finetuned.pt 0.5 out.pt
"""
import sys

import torch

old, new, a, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
so, sn = torch.load(old, map_location="cpu")["model"], torch.load(new, map_location="cpu")
it = sn.get("iter")
sn = sn["model"]
mix = {}
for k, v in sn.items():
    if k in so and so[k].shape == v.shape and v.is_floating_point():
        mix[k] = (1 - a) * so[k] + a * v
    else:
        mix[k] = v
torch.save({"model": mix, "iter": f"wise{a}_{it}"}, out)
print(f"wrote {out}: {a:.2f} x {new} (iter {it}) + {1 - a:.2f} x {old}")
