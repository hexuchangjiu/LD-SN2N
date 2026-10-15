"""Package tracked sources and explicit artifacts without recursively including old releases."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

try:
    from .install import JAR_NAME, VERSION, check_jar, is_link, safe_relative_name
except ImportError:
    from install import JAR_NAME, VERSION, check_jar, is_link, safe_relative_name

ROOT = Path(__file__).resolve().parents[1]

def digest(data):return hashlib.sha256(data).hexdigest()


def source_files(module_root: Path = ROOT) -> tuple[Path, dict[str, Path]]:
    """Resolve git's paths against the Git root, not this nested source module."""
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'],
                                       cwd=module_root, text=True).strip()).resolve()
    names = subprocess.check_output(['git', 'ls-files', '--full-name', '-z'], cwd=root).decode('utf-8').split('\0')
    module_prefix = module_root.resolve().relative_to(root).as_posix()
    exclusions = ['releases/', 'plugin/', 'dist/', 'build/']
    if module_prefix != '.':
        exclusions.extend([module_prefix + '/releases/', module_prefix + '/dist/', module_prefix + '/build/'])
    files = {}
    for name in names:
        if not name or any(name.startswith(prefix) for prefix in exclusions):
            continue
        safe_relative_name(name)
        if (Path(name).suffix.lower() in {'.pt', '.pth', '.tif', '.tiff', '.pkl', '.jar', '.zip', '.whl'}
                or any(part in {'.git', 'outputs', '.venv', '__pycache__', '.pytest_cache'} for part in Path(name).parts)):
            raise ValueError('Unexpected research/binary/cache asset in source tree: ' + name)
        source = root / name
        if is_link(source) or not source.is_file():
            raise ValueError('Tracked release source is not a regular file: ' + name)
        if any(is_link(parent) for parent in source.parents if parent != root):
            raise ValueError('Tracked release source has a link parent: ' + name)
        files[name] = source
    return root, files


def archive(path,files):
    for relative,source in files.items():
        safe_relative_name(relative)
        if is_link(source) or not source.is_file():raise ValueError('Only regular release files are packaged')
    records=[]
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as output:
        for relative,source in sorted(files.items()):
            data=source.read_bytes();name='LD-SN2N-'+VERSION+'/'+relative
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.create_system=3;info.external_attr=0o644<<16
            output.writestr(info,data)
            records.append({'path':name,'bytes':len(data),'sha256':digest(data)})
    with zipfile.ZipFile(path) as check:
        if check.testzip() is not None or any(digest(check.read(row['path'])) != row['sha256'] for row in records):
            raise ValueError('Release archive integrity check failed')
    return {'archive':path.name,'bytes':path.stat().st_size,'sha256':digest(path.read_bytes()),'files':records}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--demo-dir',type=Path)
    p.add_argument('--jar',type=Path,help='Built single JAR; defaults to repository/plugin then module/dist')
    args=p.parse_args()
    if subprocess.run(['git','diff','--quiet','HEAD'],cwd=ROOT).returncode:
        p.error('Commit source edits before packaging')
    root,files=source_files()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    jar=args.jar or root/'plugin'/JAR_NAME
    if not jar.exists() and args.jar is None:jar=ROOT/'dist'/JAR_NAME
    if not jar.exists():p.error('Build the Fiji JAR first')
    check_jar(jar,require_version=True)
    dest=args.output.resolve();dest.mkdir(parents=True,exist_ok=False)
    source=archive(dest/f'LD-SN2N-{VERSION}-source.zip',files)
    bundle=dict(files);bundle['plugin/'+jar.name]=jar
    module_prefix=ROOT.resolve().relative_to(root).as_posix()
    for wheel in (ROOT/'dist').glob('*.whl'):
        bundle[(module_prefix+'/' if module_prefix != '.' else '')+'dist/'+wheel.name]=wheel
    if args.demo_dir:
        demo=args.demo_dir.resolve(strict=True)
        metadata=json.loads((demo/'DEMO.json').read_text(encoding='utf-8'))
        if metadata.get('trained') is not False:raise ValueError('Only the explicitly non-trained synthetic fixture can be bundled')
        for n in ('DEMO.json','synthetic_static_150frames.tif','NOT_TRAINED_constant_control.pt'):
            bundle['demo_assets/'+n]=demo/n
    installation=archive(dest/f'LD-SN2N-{VERSION}-fiji-bundle.zip',bundle)
    (dest/jar.name).write_bytes(jar.read_bytes())
    manifest={'version':VERSION,'git_commit':commit,'public_license_cleared':False,
              'source':source,'installation':installation,'jar_sha256':digest(jar.read_bytes()),
              'single_jar':True,'python_dependencies_bundled':False}
    (dest/'release_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(dest),'git_commit':commit,'source_files':len(files),
        'source_zip_bytes':source['bytes'],'bundle_zip_bytes':installation['bytes'],'verified':True}))

if __name__=='__main__':main()
