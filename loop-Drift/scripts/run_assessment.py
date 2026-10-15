"""Run the same backend as Fiji with explicit files (optional CLI entry point)."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stack',required=True,type=Path)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--static-repeats',required=True,action='store_true',help='Explicitly confirm same-scene static repeats, not time dynamics/Z/SIM phases')
    p.add_argument('--query-index',type=int,default=0,help='Zero-based index')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--trust-legacy-checkpoint',action='store_true')
    p.add_argument('--offset',type=float)
    p.add_argument('--scale',type=float)
    args=p.parse_args()
    if (args.offset is None)!=(args.scale is None):p.error('Provide both --offset and --scale, or neither')
    stack=args.stack.resolve(strict=True);checkpoint=args.checkpoint.resolve(strict=True)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    request=dict(schema_version=1,domain='static_repeats',stack_path=str(stack),checkpoint_path=str(checkpoint),
        output_dir=str(out),query_index=args.query_index,device=args.device,trust_checkpoint=args.trust_legacy_checkpoint)
    if args.offset is not None:request['chart']={'offset':args.offset,'scale':args.scale}
    (out/'request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
    env=os.environ.copy();cache=out/'.cache';cache.mkdir()
    for key in ['TEMP','TMP','TMPDIR','MPLCONFIGDIR']:env[key]=str(cache)
    env['PYTHONDONTWRITEBYTECODE']='1'
    if args.trust_legacy_checkpoint:print('Legacy pickle explicitly enabled. Use only a file you control.',flush=True)
    result=subprocess.run([sys.executable,str(ROOT/'fiji_calibrated_plugin/backend/bridge.py'),
        '--request',str(out/'request.json'),'--response',str(out/'response.json')],env=env)
    if (out/'response.json').exists():print((out/'response.json').read_text(encoding='utf-8'))
    return result.returncode

if __name__=='__main__':raise SystemExit(main())
