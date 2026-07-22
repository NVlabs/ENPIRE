#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse,json,os; from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import numpy as np
def episodes(p):
    p=Path(p).expanduser(); return [p] if (p/"action-source.json").exists() else [x for x in sorted(p.iterdir()) if (x/"action-source.json").exists()]
def seg_ends(src):
    out=[]; last=None
    for i in range(len(src)+1):
        cur="__end__" if i==len(src) else ("rl" if str(src[i]).lower() in ("rl","policy") else str(src[i]).lower())
        if i and cur!=last and last in ("human","rl"): out.append(i-1)
        last=cur
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("path",nargs="?",default=os.environ.get("YAM_RAW_PATH","")); ap.add_argument("--out",default="reward_z_height.png"); a=ap.parse_args(); good=[]; bad=[]
    for ep in episodes(a.path):
        s,r=ep/"state-eef-rot6d.npy",ep/"reward.npy"
        if not s.exists() or not r.exists(): continue
        state,rew,src=np.load(s),np.load(r),json.loads((ep/"action-source.json").read_text())
        if (ep/"reward.npz").exists():
            with np.load(ep/"reward.npz") as z:
                if "rewards" in z.files and not np.array_equal(z["rewards"],rew): raise ValueError(f"reward mismatch: {ep}")
        for i in seg_ends(src): (good if np.isclose(rew[i],1.0) else bad).extend(map(float,state[i,[2,12]] if state.shape[1]>12 else state[i,[2]]))
    plt.figure(figsize=(8,4.5))
    if bad: plt.hist(bad,bins=40,density=True,alpha=.45,color="#7f1d1d",label=f"fail n={len(bad)}")
    if good: plt.hist(good,bins=40,density=True,alpha=.45,color="#166534",label=f"success n={len(good)}")
    plt.xlabel("EEF z height"); plt.ylabel("density"); plt.legend(); plt.tight_layout(); plt.savefig(a.out,dpi=160); print(a.out)
if __name__=="__main__": main()
