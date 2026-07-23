# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import urllib.parse as up
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HTML=r"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Reward Relabel</title>
<style>body{margin:0;background:#f7f7f4;color:#171717;font:14px ui-sans-serif,system-ui,sans-serif}header{position:sticky;top:0;z-index:2;display:flex;gap:10px;align-items:center;padding:12px 16px;background:#fffffc;border-bottom:1px solid #ddd}main{padding:18px;max-width:1320px;margin:auto}.brand{font-weight:700}input{width:min(46vw,620px)}input,select,button{font:inherit;border:1px solid #c9c9c2;border-radius:7px;background:white;padding:8px 10px}button{cursor:pointer}.nav{margin-left:auto;display:flex;gap:8px;align-items:center}.score{font-size:18px;font-weight:800;margin:0 0 14px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.col{min-width:0}.col h2{font-size:15px;margin:0 0 10px}.card{background:white;border:1px solid #deded8;border-radius:8px;padding:10px;margin:0 0 12px;box-shadow:0 1px 2px #0000000a}.top{display:flex;gap:8px;align-items:center;justify-content:space-between;margin-bottom:8px}.tag{font-size:12px;font-weight:700;border-radius:999px;padding:3px 8px;background:#ece9e1;color:#111}.meta{color:#666;font-size:12px}.imgs{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}.shot{min-width:0}.shot img,.shot video{width:100%;aspect-ratio:1.35;object-fit:cover;background:#111;border-radius:6px}.cap{margin-top:3px;color:#666;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.actions{display:flex;gap:8px;margin-top:9px}.ok,.bad{border:0;color:white;font-weight:700;flex:1}.rv1{color:#166534}.rv0{color:#7f1d1d}.ok{background:#166534;color:white}.bad{background:#7f1d1d;color:white}.toast{position:fixed;right:18px;bottom:18px;background:#18181b;color:white;border-radius:8px;padding:10px 12px;opacity:0;transform:translateY(8px);transition:.18s}.toast.show{opacity:1;transform:none}@media(max-width:850px){header{flex-wrap:wrap}.grid{grid-template-columns:1fr}input{width:100%}.nav{margin-left:0}}</style></head>
<body><header><span class=brand>Reward Relabel</span><input id=root placeholder="YAM_RAW_PATH or gearraw folder"><button onclick=load()>Load</button><select id=ep onchange=load(this.value)></select><input id=clip type=number min=0 step=.1 value=0 title="clip seconds, 0 = last frame" onchange=render() style="width:72px"><span id=info class=meta></span><div class=nav><button onclick=page(-1)>Prev</button><span id=pg class=meta></span><button onclick=page(1)>Next</button><input id=jump type=number min=1 title="episode number" style="width:62px"><button onclick=go()>Go</button></div></header><main><div id=score class=score></div><div class=grid><section class="col human"><h2>Human segment ends</h2><div id=human></div></section><section class="col rl"><h2>RL segment ends</h2><div id=rl></div></section></div></main><div id=toast class=toast></div>
<script>
let D=null,P=0,N=5;const $=id=>document.getElementById(id),qs=o=>new URLSearchParams(o);$("root").value=localStorage.rewardRoot||__ROOT__;$("clip").value=0;
async function api(url,opt){let r=await fetch(url,opt);if(!r.ok)throw Error(await r.text());return r.headers.get("content-type")?.includes("json")?r.json():r.text()}
async function load(ep,p=0){try{let root=$("root").value.trim();localStorage.rewardRoot=root;D=await api("/data?"+qs({root,ep:ep||$("ep").value}));P=p;$("clip").value=0;$("ep").innerHTML=D.episodes.map(e=>`<option ${e==D.episode?"selected":""}>${e}</option>`).join("");P=Math.min(P,maxp());render()}catch(e){alert(e)}}
function side(s){return D?D.segments.filter(x=>x.source==s):[]}function maxp(){return Math.max(0,Math.ceil(Math.max(side("human").length,side("rl").length)/N)-1)}function ei(){return D?D.episodes.indexOf(D.episode):-1}function page(d){$("clip").value=0;if(!D)return;let n=P+d,i=ei(),m=maxp();if(n>m&&i<D.episodes.length-1)return load(D.episodes[i+1],0);if(n<0&&i>0)return load(D.episodes[i-1],1e9);P=Math.max(0,Math.min(m,n));render()}
async function go(){if(!D)await load();let n=+$("jump").value;if(!D||!Number.isFinite(n)||n<1||n>D.episodes.length)return alert(`episode must be 1-${D?.episodes?.length||0}`);load(D.episodes[n-1],0)}function render(){let h=side("human"),r=side("rl");$("jump").max=D.episodes.length;$("score").textContent=`${ei()+1}/${D.episodes.length} episodes | ${D.success_count} successes within this episode`;$("human").innerHTML=cards(h.slice(P*N,P*N+N));$("rl").innerHTML=cards(r.slice(P*N,P*N+N));$("info").textContent=`${D.episode} | ${D.steps} steps | ${D.cameras.length} cameras | ${D.segments.length} segment ends`;$("pg").textContent=`${P+1}/${maxp()+1}`}
function cards(a){return a.length?a.map(card).join(""):`<div class=meta>None on this page</div>`}
function imgs(s){let b={root:D.root,ep:D.episode},d=Math.max(0,+$("clip").value||0);return D.cameras.map(c=>{let i=s.frames[c],u=(d?"/clip?":"/frame?")+qs({...b,cam:c,i,dur:d}),m=d?`<video src="${u}" muted autoplay loop controls playsinline></video>`:`<img src="${u}" title="${c} frame ${i}">`;return `<div class=shot>${m}<div class=cap>${c} · frame ${i}${d?` · ${d}s`:``}</div></div>`}).join("")}
function card(s){let rc=+s.reward?"rv1":"rv0";return `<article class="card ${s.source}"><div class=top><span><span class=tag>${s.source}</span> <b class=${rc}>Reward ${s.reward}</b></span><span class=meta>step ${s.idx} | t ${s.t.toFixed(3)}s</span></div><div class=imgs>${imgs(s)}</div><div class=actions><button class=ok onclick="setr(${s.idx},1)">Success</button><button class=bad onclick="setr(${s.idx},0)">Fail</button></div></article>`}
async function setr(idx,value){let r=await api("/set",{method:"POST",body:JSON.stringify({root:D.root,ep:D.episode,idx,value})});D.success_count=r.success_count;D.segments.filter(s=>s.idx==idx).forEach(s=>s.reward=value.toFixed(1));toast("Reward override completed");render()}
function toast(t){let x=$("toast");x.textContent=t;x.classList.add("show");setTimeout(()=>x.classList.remove("show"),1600)}
</script></body></html>"""
def is_ep(p): return (p/"action-source.json").is_file()
def epdir(root, ep): r=Path(root).expanduser().resolve(); return r if is_ep(r) else r/ep
def episodes(root):
    r=Path(root).expanduser().resolve()
    return [r.name] if is_ep(r) else [p.name for p in sorted(r.iterdir()) if p.is_dir() and is_ep(p)]
def norm(s): s=str(s).lower(); return "human" if s=="human" else ("rl" if s in ("rl","policy") else s)
def cameras(d): return [p.name[:-15] for p in sorted(d.glob("*-images-rgb.mp4"))]
def frame_idx(comp, i, key):
    if i>=len(comp) or key not in comp[i]: return i
    t=comp[i][key]; vals=[(abs(row[key]-t),j) for j,row in enumerate(comp) if key in row]
    return min(vals)[1] if vals else i
def load_episode(root, ep=""):
    es=episodes(root)
    if not es: raise FileNotFoundError("No episode folders with action-source.json found")
    ep=ep if ep in es else es[0]; d=epdir(root, ep); cams=cameras(d)
    if not cams: raise FileNotFoundError(f"No *-images-rgb.mp4 files found in {d}")
    src=json.loads((d/"action-source.json").read_text())
    comp=json.loads((d/"component_timestamps.json").read_text()) if (d/"component_timestamps.json").exists() else [{} for _ in src]
    ts=np.load(d/"timestamp.npy") if (d/"timestamp.npy").exists() else np.arange(len(src))
    rew=np.load(d/"reward.npy") if (d/"reward.npy").exists() else np.zeros(len(src), dtype=np.float32)
    if (d/"reward.npz").exists():
        with np.load(d/"reward.npz") as z:
            if "rewards" in z.files and not np.array_equal(z["rewards"],rew): raise ValueError("reward.npy and reward.npz rewards mismatch")
    segs=[]; last=None
    for i in range(len(src)+1):
        cur="__end__" if i==len(src) else norm(src[i])
        if i and cur!=last and last in ("human","rl"):
            k=i-1; t=float(ts[k]-ts[0]) if len(ts)>k else float(k); frames={c:frame_idx(comp,k,c) for c in cams}
            segs.append(dict(idx=k,source=last,tag=str(src[k]),t=t,reward=f"{float(rew[k]):.1f}",frames=frames))
        last=cur
    return dict(root=str(Path(root).expanduser().resolve()),episode=ep,episodes=es,steps=len(src),cameras=cams,segments=segs,success_count=int(np.sum(np.isclose(rew,1.0))))
def read_frame(root, ep, cam, i):
    import cv2
    f=epdir(root, ep)/f"{cam}-images-rgb.mp4"; cap=cv2.VideoCapture(str(f)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok,img=cap.read(); cap.release()
    if not ok: raise FileNotFoundError(f"Could not read {f} frame {i}")
    ok,buf=cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
    if not ok: raise RuntimeError("JPEG encode failed")
    return buf.tobytes()
def read_clip(root, ep, cam, i, dur):
    import os
    import subprocess
    import tempfile

    import cv2
    f=epdir(root, ep)/f"{cam}-images-rgb.mp4"; cap=cv2.VideoCapture(str(f)); fps=cap.get(cv2.CAP_PROP_FPS) or 30; cap.release(); d=max(.001,float(dur)); st=max(0,(int(i)+1)/fps-d)
    tmp=tempfile.NamedTemporaryFile(suffix=".mp4",delete=False); name=tmp.name; tmp.close()
    cmd=["ffmpeg","-hide_banner","-loglevel","error","-y","-ss",str(st),"-t",str(d),"-i",str(f),"-an","-vf","format=yuv420p","-c:v","libx264","-preset","ultrafast","-movflags","+faststart",name]
    subprocess.run(cmd,check=True); data=Path(name).read_bytes(); os.unlink(name); return data
def set_reward(root, ep, idx, value):
    d=epdir(root, ep); idx=int(idx); value=float(value); rpath=d/"reward.npy"
    rewards=np.array(np.load(rpath))
    if value not in (0.0,1.0) or idx<0 or idx>=len(rewards): raise ValueError(f"reward index/value out of range: idx={idx}, value={value}")
    rewards[idx]=value; np.save(rpath, rewards)
    data={}
    if (d/"reward.npz").exists():
        with np.load(d/"reward.npz") as z: data={k:np.array(z[k]) for k in z.files}
    elif (d/"dones.npy").exists(): data["dones"]=np.load(d/"dones.npy")
    for k in [k for k in ("rewards","reward") if k in data] or ["rewards"]:
        a=np.array(data.get(k, rewards)); a[idx]=value; data[k]=a
    np.savez(d/"reward.npz", **data); return {"ok": True, "success_count": int(np.sum(np.isclose(rewards,1.0)))}
class Handler(BaseHTTPRequestHandler):
    def out(self, body, typ="application/json", code=200):
        body=body if isinstance(body, bytes) else body.encode(); self.send_response(code); self.send_header("Content-Type", typ); self.send_header("Cache-Control","no-store"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        try:
            u=up.urlparse(self.path); q={k:v[0] for k,v in up.parse_qs(u.query).items()}
            if u.path=="/": return self.out(HTML.replace("__ROOT__",json.dumps(os.environ.get("YAM_RAW_PATH",""))), "text/html")
            if u.path=="/data": return self.out(json.dumps(load_episode(q.get("root",""), q.get("ep",""))))
            if u.path=="/frame": return self.out(read_frame(q["root"], q["ep"], q["cam"], q["i"]), "image/jpeg")
            if u.path=="/clip": return self.out(read_clip(q["root"], q["ep"], q["cam"], q["i"], q["dur"]), "video/mp4")
            self.out("not found", "text/plain", 404)
        except Exception as e: self.out(str(e), "text/plain", 500)
    def do_POST(self):
        try:
            b=json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))) or b"{}")
            self.out(json.dumps(set_reward(b["root"], b["ep"], b["idx"], b["value"]))) if self.path=="/set" else self.out("not found","text/plain",404)
        except (ValueError,IndexError,KeyError) as e: self.out(str(e), "text/plain", 400)
        except Exception as e: self.out(str(e), "text/plain", 500)
if __name__=="__main__":
    port=8767
    while True:
        try: srv=ThreadingHTTPServer(("127.0.0.1", port), Handler); break
        except OSError: port+=1
    url=f"http://127.0.0.1:{port}"; print(url, flush=True); webbrowser.open(url); srv.serve_forever()
