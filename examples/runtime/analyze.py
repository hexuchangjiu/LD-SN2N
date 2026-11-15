"""Full-field E95 simulation: ordered FP32 256/128 tile inference and Gaussian feedback.

Portable released backend; the exact input/model/perturbation identity is verified.
Every state is evaluated at native 2048 resolution. A completed receipt is written last.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import tifffile
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from parameters import load_model

MODEL_SHA = "d126ce3f37f6d5b44f5dc5ed1f1ead1b4bc95e45212eae8a693c131246e8c1c2"
IMAGE_SHA = "6e5525ca9296a6c35968374fbe77d03f84240f981e2e36f9cb1d2a23ea3b18fe"
SIGMA, SEED = 0.10512460972397199, 3780359674982640794
H = W = 2048

def sha_array(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast("B")).hexdigest()

def digest_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for buf in iter(lambda: f.read(1024*1024), b""):
            h.update(buf)
    return h.hexdigest()

def verified_bytes(p, expected, maximum):
    if not p.is_file() or not 0 < p.stat().st_size <= maximum:
        raise ValueError("File absent or exceeds release bound: " + p.name)
    b=p.read_bytes()
    if hashlib.sha256(b).hexdigest()!=expected:
        raise ValueError("File identity mismatch: "+p.name)
    return b

def coordinate_plane(source):
    with tifffile.TiffFile(io.BytesIO(source)) as tf:
        if len(tf.pages)!=1 or tf.pages[0].shape!=(H,W) or tf.pages[0].dtype!=np.uint8:
            raise ValueError("Expected one genuine 2048 by 2048 uint8 source plane")
        return tf.asarray().astype(np.float32)/np.float32(255)

def run(args, image_sha=IMAGE_SHA):
    t0=time.perf_counter()
    if args.profile!="e95-direct" or (args.image_sha,args.model_sha)!=(image_sha,MODEL_SHA):
        raise ValueError("Unregistered simulation identity")
    if (args.sigma,args.seed)!=(SIGMA,SEED):
        raise ValueError("Not the registered simulation challenge")
    if not 1<=args.threads<=12:
        raise ValueError("Invalid thread count")
    if args.out.exists():
        raise FileExistsError(args.out)
    if shutil.disk_usage(args.out.parent).free < 21*1024**3:
        raise OSError("Insufficient space: keep 20 GiB reserve")
    qs=coordinate_plane(verified_bytes(args.image,image_sha,8*1024**2))
    weights=verified_bytes(args.model,MODEL_SHA,128*1024**2)
    if args.device=="cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model=load_model(io.BytesIO(weights),map_location=args.device).eval()
    del weights
    args.out.mkdir(exist_ok=False)
    # Original E54/E97 Gaussian blend; same FP64 accumulation and raster tile order.
    axis=np.arange(256,dtype=np.float64)-127.5
    w1=np.exp(-.5*(axis/64.)**2)
    window=np.outer(w1,w1)
    pos=[(y,x) for y in range(0,H-255,128) for x in range(0,W-255,128)]
    denom=np.zeros((H,W),np.float64)
    for y,x in pos:
        denom[y:y+256,x:x+256]+=window
    if not np.all(denom>0): raise ValueError("Uncovered native pixels")
    batch=4 if args.device=="cuda" else 1
    calc_calls=0
    def denoise(q):
        nonlocal calc_calls
        acc=np.zeros_like(denom)
        for start in range(0,len(pos),batch):
            gp=pos[start:start+batch]
            inp=np.stack([q[y:y+256,x:x+256] for y,x in gp])[:,None]
            ts=model(torch.from_numpy(inp).to(args.device)).cpu()
            if tuple(ts.shape)!=tuple(inp.shape) or ts.dtype!=torch.float32:
                raise ValueError("Invalid denoiser output")
            ar=ts[:,0].numpy()
            for (y,x),p in zip(gp,ar):
                acc[y:y+256,x:x+256]+=window*p
        img=(acc/denom).astype(np.float32)
        if not np.isfinite(img).all(): raise ValueError("Nonfinite state")
        calc_calls+=1
        print("Native field %d/11 complete" % calc_calls,flush=True)
        return img
    st=np.memmap(args.out/"states.f32",dtype="<f4",mode="w+",shape=(5,3,H,W))
    first=np.memmap(args.out/"stream1_input_states.f32",dtype="<f4",mode="w+",shape=(6,H,W))
    rng=np.random.default_rng(args.seed)
    nh=hashlib.sha256()
    first[0]=qs
    with torch.inference_mode():
        d1=denoise(qs)
        first[1]=d1
        st[:,0]=d1
        for k in range(5):
            state=d1
            for t in (1,2):
                noise=rng.normal(0,args.sigma,(H,W)).astype("<f4")
                nh.update(memoryview(noise).cast("B"))
                actual=state+noise
                if k==0: first[2*t]=actual
                state=denoise(actual)
                st[k,t]=state
                if k==0: first[2*t+1]=state
    st.flush(); first.flush()
    if calc_calls!=11: raise AssertionError("Incomplete recursion")
    receipt=dict(status="COMPLETE",profile=args.profile,model_sha256=MODEL_SHA,image_sha256=image_sha,
                 input_shape=[H,W],input_dtype="uint8",source_to_model="uint8 / 255 (no normalization, no clipping)",
                 sigma=args.sigma,seed=args.seed,K=5,T=3,image_calls=calc_calls,tile_forwards=calc_calls*len(pos),
                 tile_size=256,tile_stride=128,tile_blend="Gaussian std=64, FP64 ordered accumulation",
                 fp32=True,states_shape=list(st.shape),states_dtype="<f4",states_sha256=digest_file(args.out/"states.f32"),
                 stream1_input_states_shape=list(first.shape),stream1_input_states_sha256=digest_file(args.out/"stream1_input_states.f32"),
                 epsilon_sha256=nh.hexdigest(),d1_sha256=sha_array(d1),input_model_chart_sha256=sha_array(qs),
                 threads=args.threads,device=args.device,torch=torch.__version__,numpy=np.__version__,
                 tifffile=tifffile.__version__,seconds=time.perf_counter()-t0)
    with (args.out/"run.json").open("x",encoding="utf8") as f: json.dump(receipt,f,indent=2,allow_nan=False)
    print("COMPLETE in %.2f s"%receipt["seconds"],flush=True)

def cli():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for opt in ("model","image","out"): p.add_argument("--"+opt,required=True,type=Path)
    for opt in ("profile","model-sha","image-sha"): p.add_argument("--"+opt,required=True)
    p.add_argument("--sigma",type=float,required=True); p.add_argument("--seed",type=int,required=True)
    p.add_argument("--threads",type=int,default=4); p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    return p.parse_args()

if __name__=="__main__":
    run(cli())
