"""Real phase-space Loop: measured BioSR source + true denoiser + calibrated raw camera + frozen fairSIM.
No random value is added to an image in reconstruction/model coordinates.
CLI refuses unregistered sources/models, camera failures, and non-new run paths.
"""
import argparse, csv, hashlib, io, json, os, shutil, struct, subprocess, sys, time, traceback
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.ndimage import gaussian_filter, percentile_filter, zoom
import scipy
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from parameters import load_model

RAW = '0aaf8d8ae16a8e46eabdd86dd01e182a4053aae5e56fcb46781eea54bf564b14'
PIN = {'fairSIM_plugin.jar':'ce221779b88cda8b071d8a95908f21d4970b30980f107ca923df2f8d2249a956',
       'FairSimBridge.class':'152d1c3ee0e196b73f15a312f5ecb4d0facf48b2fe048a061a0d524b1aa950cf',
       'TIRF_OTF2D_488.mrc':'e496e544605e9b4a702af028b1cdd946be4ee24e6ba62e845475fda06a8a363a'}
N=502; M=1004; SEED=143148
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def h(x):
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()
def write_json(p,x):
    with p.open('x',encoding='utf8') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def finite(x,shape):
    x=np.asarray(x)
    if x.shape!=shape or not np.isfinite(x).all(): raise ValueError(f'Invalid array shape or nonfinite values: {x.shape}')
    return x
def checked(p,digest):
    if sha(p)!=digest: raise ValueError('File identity mismatch: '+str(p))
def stdstat(x):
    return [float(x.min()),float(x.mean(dtype='f8')),float(x.max())]
def preprocess(y,lo,span):
    x=(y-lo)/span
    g=gaussian_filter(x,1.,mode='reflect',truncate=4.)
    coarse=g[::4,::4][8:-8,8:-8]
    bg=np.pad(gaussian_filter(percentile_filter(coarse,20.,size=17,mode='reflect'),3.,mode='reflect'),8,mode='edge')
    bg=zoom(bg,(M/bg.shape[0],M/bg.shape[1]),order=1,mode='reflect',prefilter=False)
    return finite(np.maximum(g-bg,0).astype('<f4'),(M,M))
def predict(model,x,device):
    core,stride,halo=384,256,64
    grid=sorted(set(list(range(0,M-core+1,stride))+[M-core]))
    axis=np.arange(core,dtype=np.float32)-(core-1)/2
    blend=np.exp(-(axis[:,None]**2+axis[None,:]**2)/(2*(core/4)**2))
    padded=np.pad(x,halo,mode='reflect')
    out=np.zeros_like(x);den=np.zeros_like(x)
    with torch.inference_mode():
        for y in grid:
            for z in grid:
                tile=np.ascontiguousarray(padded[y:y+core+2*halo,z:z+core+2*halo])
                inp=torch.from_numpy(tile).view(1,1,*tile.shape).to(device)
                pred=model(inp)
                if pred.shape!=inp.shape or pred.dtype!=torch.float32: raise RuntimeError('Model plane contract')
                out[y:y+core,z:z+core]+=pred[0,0].cpu().numpy()[halo:halo+core,halo:halo+core]*blend
                den[y:y+core,z:z+core]+=blend
    assert len(grid)==4 and np.all(den>0)
    return finite((out/den).astype('<f4'),(M,M))
def recon(raw,out,kit,java,params=None):
    out.mkdir(exist_ok=False)
    # Keep all raw captures and intermediate geometries so that reconstruction is auditable.
    finite(raw,(9,N,N)).astype('<f4').tofile(out/'phases.f32')
    props={'order':'phase_fast','wiener':0.05,'pixel_um':0.0626,'fade':10,'apo_cutoff':2,'apo_bend':0.9,
           'na':1.3,'emission_nm':525,'size':N,'input':(out/'phases.f32').as_posix(),'output':out.as_posix(),
           'otf_radial_mrc':(kit/'TIRF_OTF2D_488.mrc').as_posix()}
    if params is not None: props['frozen_params']=params.as_posix()
    def escaped(v):
        # Java Properties.load uses Latin-1 and unicode escapes (utf-16 units).
        return ''.join(chr(c) if 32<=c<127 else '\\u%04x'%c for c in np.frombuffer(str(v).encode('utf-16-le'),dtype='<u2'))
    (out/'job.properties').write_text('\n'.join(k+'='+escaped(v) for k,v in props.items()),encoding='ascii')
    cmd=[str(java),'-Djava.awt.headless=true','-XX:ActiveProcessorCount=2','-Xmx1700m','-cp',str(kit)+os.pathsep+str(kit/'fairSIM_plugin.jar'),'FairSimBridge',str(out/'job.properties')]
    with (out/'recon.stdout.log').open('xb') as f,(out/'recon.stderr.log').open('xb') as e:
        subprocess.run(cmd,check=True,cwd=out,timeout=300,stdout=f,stderr=e)
    return finite(np.fromfile(out/'reconstruction.f32',dtype='<f4').reshape(M,M),(M,M))
def freeze_illumination(base,target):
    # Restore higher precision CSV entries instead of fairSIM XML's six-digit decimal writer.
    cfg=ET.parse(base/'parameters.xml')
    rows=list(csv.DictReader((base/'illumination.csv').read_text(encoding='utf8').splitlines()))
    assert len(rows)==3
    for i,r in enumerate(rows):
        d=cfg.find(f'.//sim-param/dir-{i}'); assert int(r['angle'])==i
        for key,vs in [('shift',[r['kx_pixel'],r['ky_pixel']]),('phase-offset',[r['phase_rad']]),('modulations',['1',r['raw_modulation']])]:
            q=d.find(key);assert np.allclose(np.fromstring(q.text,sep=' '),np.array(vs,dtype=float),atol=1e-5,rtol=7e-6)
            q.text=' '+' '.join(vs)+' '
    cfg.write(target,encoding='utf-8',xml_declaration=True)
def rng(k,t):return np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED,k,t])))
def arg_parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for k in ('raw','model','assets','out','java'):p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--preflight',action='store_true',help='No stochastic draws or feedback: verify measured roundtrip, frozen reconstruction and D1')
    return p
def run(a,began):
    kit=a.assets/'recon_assets'; d=a.out
    a.native_d1=a.assets/'native_D1.f32'
    a.native_d1_sha='c2d76dfd0c92cc5e77bdbfff7324db174de6f5f91f08158d8b61081c1af0d015'
    a.model_sha='c5e1d221f9f84a5fda53f129d493901f7626e07b38e07a0249815f238a09f4f5'
    a.native_checkpoint='MT exposure-paired terminal checkpoint (training provenance; not needed for inference)'
    a.native_checkpoint_sha='a1c180fd7e040c5dbe06cf25f18fa2653d43cfe6218db85237609f8d03f3f4b4'
    a.conversion_dir=a.assets/'counts'
    checked(a.conversion_dir/'calibration.json','3ea412c904167a72cb4d63ec0229467166bfed92ea2f2938b2d9cb3b84e6ce0c')
    for p,ref in PIN.items():checked(kit/p,ref)
    checked(a.raw,RAW);checked(a.model,a.model_sha);checked(a.native_d1,a.native_d1_sha)
    if shutil.disk_usage(d).free < 21*2**30:raise OSError('At least 21 GiB of free disk required')
    rawbytes=a.raw.read_bytes(); assert struct.unpack_from('<4i',rawbytes)==(N,N,9,6) and struct.unpack_from('<i',rawbytes,92)[0]==0
    raw=np.frombuffer(rawbytes,dtype='<u2',offset=1024).reshape(9,N,N).astype('<f4')
    if not a.preflight:
        checked(a.assets/'camera_qa.json','70cc669875e73162f7b9a2bd6a934dd6f856f15d49ed58cd28cdac3a3c5f68e5')
        cq=json.loads((a.assets/'camera_qa.json').read_text(encoding='utf8'))
        checked(a.assets/'camera.py',cq['camera_sha256'])
        sys.path.insert(0,str(a.assets))
        from camera import Camera
        if cq.get('status')!='PASS' or len(cq['gates'])!=9 or not all(v is True for v in cq['gates'].values()):raise RuntimeError('raw-camera qualification did not pass')
        cam_path=a.conversion_dir/'camera_model.npz'
        cr=json.loads((a.conversion_dir/'calibration.json').read_text(encoding='utf8'))
        assert cr['status']=='PASS' and cr['source_d1_sha']==a.native_d1_sha
        assert cr['source_model_sha']=='f11e8d3819ecf2aba9a92b056fcb9f2376171c890bf336e037bd439a090b4c04'
        assert cq['npz_sha256']==cr['source_camera_sha']
        checked(cam_path,cr['new_camera_sha'])
        with np.load(cam_path,allow_pickle=False) as new:
            for key,base in cr['before_scale'].items():
                np.testing.assert_array_equal(new[key],np.asarray(base,dtype=np.float32)*np.float32(cr['scalar_gain']))
        cam=Camera(cam_path)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=load_model(a.model,map_location=a.device).eval()
    start=time.perf_counter();base=d/'measured'
    z=recon(raw,base,kit,a.java);lo,span=float(z.min()),float(z.max()-z.min())
    xin=preprocess(z,lo,span);xin.tofile(d/'input.f32')
    d1=predict(model,xin,a.device)
    if h(xin)!='4745fa16d42e33a36e2fd98628bbbc88bec2bc1a7c3f938c16c334f12511917b':
        raise RuntimeError('The measured reconstruction/input chart does not match the registered acquisition')
    ref=np.fromfile(a.native_d1,dtype='<f4').reshape(M,M)
    d1err=float(np.max(abs(ref-d1)))
    assert d1err < 2e-5, ('Model parity',d1err)
    freeze_illumination(base,d/'frozen.xml')
    test=recon(raw,d/'measured_frozen_replay',kit,a.java,d/'frozen.xml')
    rr=float(np.max(abs(z-test)))/span
    assert rr<1e-5,('Frozen geometric reconstruction changed measured input',rr)
    init=dict(status='PASS_MEASURED_INPUT',raw_sha256=RAW,model_sha256=a.model_sha,native_checkpoint=str(a.native_checkpoint),native_checkpoint_sha256=a.native_checkpoint_sha,device=a.device,torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,shape=list(raw.shape),
              chart_min=lo,chart_span=span,actual_D1_sha256=h(d1),reference_D1_sha256=a.native_d1_sha,actual_D1_max_abs_reference=d1err,
              frozen_recon_max_abs_over_chart_span=rr,frozen_geometry_sha256=sha(d/'frozen.xml'),frozen_recon_illumination_sha256=sha(d/'measured_frozen_replay/illumination.csv'),
              measured_elapsed_s=time.perf_counter()-start,preprocessing='Original measured reconstruction min/range fixed at every step. Same deterministic gaussian/background/zero-floor pipeline per reconstructed image; no raw-noise injection in this step.',
              recon_files_sha256=PIN)
    write_json(d/'measured_acceptance.json',init)
    if a.preflight:
        print('MEASURED PREFLIGHT PASS',init,flush=True);return
    print('D1 accepted, starting raw feedback. Max abs parity error:',d1err,flush=True)
    states=np.empty((5,3,M,M),dtype='<f4');states[:,0]=d1
    ins=np.empty((5,2,M,M),dtype='<f4');rrs=[];last_ih=init['frozen_recon_illumination_sha256']
    mu0=finite(cam.expected(d1),(9,N,N)); assert mu0.min()>=0 and mu0.max()>0
    from camera_transfer import qualify
    transfer=qualify(cq,cam,d1,raw)
    transfer.update(refit=a.conversion_dir is not None,actual_camera=str(cam_path),camera_sha=sha(cam_path),scalar_gain=None if not a.conversion_dir else cr['scalar_gain'])
    write_json(d/'camera_transfer.json',transfer)
    if not all(transfer['gates'].values()):raise RuntimeError('Fixed original camera failed to transfer to replacement model state. See camera_transfer.json; do not continue as matched.')
    digests=set()
    for k in range(5):
        for t in [0,1]:
            start=time.perf_counter(); current=states[k,t]; sd=d/f'k{k+1}_transition{t+1}'
            mu=mu0 if t==0 else finite(cam.expected(current),(9,N,N))
            mind=float(mu.min());assert mind>=0
            rms=float(np.sqrt(np.mean((mu.astype(float)-mu0.astype(float))**2)))
            if t==1 and not rms>1e-8:raise RuntimeError('Second predicted exposure independent of previous state')
            shot=finite(cam.sample(mu,rng(k,t)),(9,N,N)).astype('<f4')
            if not np.array_equal(shot,cam.sample(mu,rng(k,t)).astype('<f4')):raise RuntimeError('raw seed replay failed')
            rd=h(shot)
            if rd in digests:raise RuntimeError('Duplicate exposure')
            digests.add(rd)
            begin=time.perf_counter();z=recon(shot,sd,kit,a.java,d/'frozen.xml');recont=time.perf_counter()-begin
            assert sha(sd/'illumination.csv')==last_ih,'Frozen geometry changed across draws'
            begin=time.perf_counter();nt=preprocess(z,lo,span);nxt=predict(model,nt,a.device);networkt=time.perf_counter()-begin
            states[k,t+1]=nxt;ins[k,t]=nt
            row=dict(stream=k+1,source_state=t+1,destination_state=t+2,parent_state_sha256=h(current),next_state_sha256=h(nxt),mu_photon_sha256=h(mu),mu_min_mean_max=stdstat(mu),raw_ADU_sha256=rd,
                     rng='PCG64(SeedSequence([root, zero-based stream, zero-based transition]))',root_seed=SEED,raw_seed_replay=True,mu_RMS_difference_from_D1_photons=rms,mu_mean_difference_from_D1_photons=float(np.mean(mu-mu0,dtype='f8')),
                     raw_min_mean_max=stdstat(shot),model_input_sha256=h(nt),state_min_mean_max=stdstat(nxt),recon_seconds=recont,preprocess_denoise_seconds=networkt,elapsed_s=time.perf_counter()-start)
            write_json(sd/'transition.json',row);rrs.append(row)
            print(f'k{k+1} {t+1}->{t+2}: {row["elapsed_s"]:.2f}s; signal delta versus D1 {rms:.4g} photons',flush=True)
    states.tofile(d/'states.f32');ins.tofile(d/'renoised_inputs.f32')
    assert len(rrs)==10 and len(digests)==10 and np.all(states[:,0]==d1)
    write_json(d/'run.json',dict(status='COMPLETE_RAW_LOOP',K=5,T=3,shape=list(states.shape),dtype='<f4',order='C [stream,state,y,x]',
             profile='Measured BioSR MT L05 + pixel-center-convention corrected empirical raw-camera re-exposure, not a published per-source physical calibration',
             actual_state_stack_sha256=sha(d/'states.f32'),actual_reconstruct_inputs_sha256=sha(d/'renoised_inputs.f32'),expected_camera_identity=sha(cam_path),
             camera_qa_identity=sha(a.assets/'camera_qa.json'),code_sha256=sha(Path(__file__)),confirmed_D1=h(d1),image_model_calls=11,tile_model_calls=176,raw_transitions=10,sampled_detector_planes=90,
             auxiliary_QA='10 seed replays not propagated; 1 deterministic frozen-geometry repeat of measured reconstruction',
             input=a.raw.name,model=a.model.name,elapsed_s=time.perf_counter()-began,transitions=rrs))
    print('COMPLETE_RAW_LOOP',d,flush=True)
if __name__=='__main__':
    if not __debug__: raise RuntimeError('Cannot disable validation via optimized Python')
    if sys.flags.isolated:
        sys.path.insert(0,str(Path(__file__).resolve().parent))
    a=arg_parser().parse_args()
    for k in ['raw','model','java','assets','out']:setattr(a,k,getattr(a,k).resolve())
    a.out.mkdir(exist_ok=False,parents=False)
    try:run(a,time.perf_counter())
    except Exception:
        (a.out/'INCOMPLETE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise

