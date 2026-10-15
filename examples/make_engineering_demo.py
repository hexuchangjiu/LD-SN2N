"""Generate non-biological fixtures, NOT a trained or effective denoiser.

The deliberately constant AUnet checks installation, tensor shapes and exports.
It is a negative control and must not be used in a paper efficacy figure.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import tifffile
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fiji_calibrated_plugin.backend.vendor.aunet import AUnet

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    y,x=np.mgrid[:64,:80]
    scene=100+30*np.exp(-.5*((x-25-5*np.sin(y/9))/2)**2)+20*np.exp(-.5*((y-35)/3)**2)
    stack=(scene+np.random.default_rng(1015).normal(0,3,(150,64,80))).astype(np.float32)
    tifffile.imwrite(out/'synthetic_static_150frames.tif',stack,metadata={'axes':'TYX'})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1015);model=AUnet(1,1,True).eval()
    with torch.no_grad():
        for param in model.parameters():param.zero_()
    torch.save({'schema':'loopdrift-engineering-negative-control-v1','model_state_dict':model.state_dict(),
        'chart':{'offset':100.,'scale':50.},'trained':False,
        'warning':'Deliberately constant output, not a trained denoiser or efficacy example'},out/'NOT_TRAINED_constant_control.pt')
    note={'purpose':'Engineering negative control only; no microscopy or denoising evidence',
        'frames':150,'shape':[64,80],'expected_native_output':125.,'trained':False,
        'run':'python scripts/run_assessment.py --stack <demo>/synthetic_static_150frames.tif --checkpoint <demo>/NOT_TRAINED_constant_control.pt --output <new-output> --static-repeats --device cpu'}
    (out/'DEMO.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
    print(json.dumps(note))

if __name__=='__main__':main()
