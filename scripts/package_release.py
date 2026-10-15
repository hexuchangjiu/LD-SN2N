"""Package only tracked release sources and explicit build artifacts, never research data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
VERSION='0.4.0'

def digest(data):return hashlib.sha256(data).hexdigest()

def archive(path,files):
    records=[]
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as output:
        for relative,source in sorted(files.items()):
            if source.is_symlink() or not source.is_file():raise ValueError('Only regular release files are packaged')
            data=source.read_bytes();name='LD-SN2N-'+VERSION+'/'+relative
            info=zipfile.ZipInfo(name,date_time=(2026,10,15,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            output.writestr(info,data)
            records.append({'path':name,'bytes':len(data),'sha256':digest(data)})
    with zipfile.ZipFile(path) as check:
        assert check.testzip() is None
        for row in records:assert digest(check.read(row['path']))==row['sha256']
    return {'archive':path.name,'bytes':path.stat().st_size,'sha256':digest(path.read_bytes()),'files':records}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--demo-dir',type=Path)
    args=p.parse_args()
    if subprocess.run(['git','diff','--quiet','HEAD'],cwd=ROOT).returncode:
        p.error('Commit source edits before packaging')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0')
    files={n:ROOT/n for n in names if n}
    for n in files:
        if Path(n).suffix.lower() in {'.pt','.pth','.tif','.tiff','.pkl'} or n.startswith(('.git/','outputs/','.venv/')):
            raise ValueError('Unexpected research/binary asset in source tree: '+n)
    jar=ROOT/f'dist/loopsn2n-calibrated-fiji-{VERSION}.jar'
    if not jar.exists():p.error('Build the Fiji JAR first')
    dest=args.output.resolve();dest.mkdir(parents=True,exist_ok=False)
    source=archive(dest/f'LD-SN2N-{VERSION}-source.zip',files)
    bundle=dict(files);bundle['dist/'+jar.name]=jar
    for wheel in (ROOT/'dist').glob('*.whl'):bundle['dist/'+wheel.name]=wheel
    if args.demo_dir:
        demo=args.demo_dir.resolve(strict=True)
        metadata=json.loads((demo/'DEMO.json').read_text())
        if metadata.get('trained') is not False:raise ValueError('Only the explicitly non-trained synthetic fixture can be bundled')
        for n in ('DEMO.json','synthetic_static_150frames.tif','NOT_TRAINED_constant_control.pt'):
            bundle['demo_assets/'+n]=demo/n
    installation=archive(dest/f'LD-SN2N-{VERSION}-fiji-bundle.zip',bundle)
    (dest/jar.name).write_bytes(jar.read_bytes())
    manifest={'version':VERSION,'git_commit':commit,'public_license_cleared':False,
              'source':source,'installation':installation,'jar_sha256':digest(jar.read_bytes())}
    (dest/'release_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(dest),'git_commit':commit,'source_files':len(files),
        'source_zip_bytes':source['bytes'],'bundle_zip_bytes':installation['bytes'],'verified':True}))

if __name__=='__main__':main()
