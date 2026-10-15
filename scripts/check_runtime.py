"""Read-only runtime preflight; no package installation or checkpoint loading."""
from __future__ import annotations
import argparse
import importlib
import json
import platform
import re
import sys

MODULES = ('numpy','scipy','torch','tifffile','skimage','matplotlib','PIL')

def check(require_cuda=False):
    errors=[]
    versions={}
    if sys.version_info < (3,10):
        errors.append('Python >=3.10 is required')
    for name in MODULES:
        try:
            module=importlib.import_module(name)
            versions[name]=getattr(module,'__version__','unknown')
        except Exception as exc:
            errors.append(f'{name}: {type(exc).__name__}: {exc}')
    cuda=False
    if 'torch' in versions:
        import torch
        version=re.match(r'^(\d+)\.(\d+)',torch.__version__)
        if not version or tuple(map(int,version.groups())) < (2,6):
            errors.append('PyTorch >=2.6 is required for supported weights-only loading')
        try:
            assert torch.isfinite(torch.ones(2).square()).all()
            cuda=torch.cuda.is_available()
            if require_cuda:
                if not cuda: raise RuntimeError('CUDA is not available in this interpreter')
                assert torch.isfinite(torch.ones(2,device='cuda').square()).all()
        except Exception as exc:
            errors.append(f'tensor check: {exc}')
    return {'ok':not errors,'python':sys.executable,'python_version':platform.python_version(),
            'platform':platform.system(),'versions':versions,'cuda_available':cuda,
            'cuda_execution_tested':bool(require_cuda and cuda and not errors),'errors':errors}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json',action='store_true')
    parser.add_argument('--require-cuda',action='store_true')
    args=parser.parse_args()
    report=check(args.require_cuda)
    print(json.dumps(report,indent=2) if args.json else
          ('Runtime OK' if report['ok'] else 'Runtime NOT READY')+'\n'+ '\n'.join(report['errors']))
    return 0 if report['ok'] else 1

if __name__=='__main__':
    raise SystemExit(main())
