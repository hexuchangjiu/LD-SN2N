"""Run software contracts and record results with source hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();cache=out/'cache';cache.mkdir()
    for k in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR'):env[k]=str(cache)
    env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONUTF8']='1'
    checks=[]
    for label,command in [
        ('runtime',[sys.executable,str(ROOT/'scripts/check_runtime.py'),'--json']),
        ('unit_contracts',[sys.executable,'-m','unittest','discover','-s','tests','-v']),
        ('algorithm_demo',[sys.executable,'-m','examples.algorithm_demo'])]:
        result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=300)
        (out/(label+'.stdout.log')).write_text(result.stdout,encoding='utf-8')
        (out/(label+'.stderr.log')).write_text(result.stderr,encoding='utf-8')
        checks.append({'name':label,'exit_code':result.returncode})
    record={'status':'PASS' if all(c['exit_code']==0 for c in checks) else 'FAIL','checks':checks,
        'science_efficacy_test':False,'gui_interactive_verified':False,
        'source_sha256':{str(f.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
                         for folder in ('fiji_calibrated_plugin/backend','loopdrift') for f in sorted((ROOT/folder).rglob('*.py'))}}
    (out/'verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2));return 0 if record['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
