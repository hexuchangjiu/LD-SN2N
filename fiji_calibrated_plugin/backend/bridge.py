"""E69: calibrated same-scene challenge, frozen operator, full-plane Map.

This research workflow cannot inherit the strict c640 deployment thresholds.
The compatibility P1--P4 readout is labelled diagnostic; formal output abstains.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key,'4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import hashlib
import html
import json
import platform
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fiji_calibrated_plugin.backend.noise_calibration import calibrate_stack, sample_noise
from fiji_calibrated_plugin.backend.vendor.energy import signature_from_Y, verdict, decide_from_energy
from fiji_calibrated_plugin.backend.vendor.loop_map import signature_map_from_Y, render_2d_colormap
from fiji_calibrated_plugin.backend.support_display import render_support_display

SCHEMA = 'loopsn2n-fiji-calibrated-v2-display1px'
K,T,SEED = 5,3,690922

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value.resolve())
    raise TypeError(type(value).__name__)

def write_json(path,obj):
    path=Path(path)
    stage=path.with_suffix(path.suffix+'.partial')
    stage.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=serial),encoding='utf-8')
    for attempt in range(6):
        try:
            os.replace(stage,path)
            break
        except PermissionError:
            if attempt==5:raise
            time.sleep(.05*(attempt+1))  # bounded Windows reader/indexer sharing retry

class Cancelled(Exception):pass

class OutputDirectoryConflict(FileExistsError):
    """Refuse a reused job without overwriting any of its existing contents."""

def validate_output_directory(out):
    allowed={'request.json','source_stack.tif','STOP_REQUESTED.json','RUNNING.lock',
             'worker.log','runtime_probe.log','.cache'}
    unexpected=[p.name for p in out.iterdir() if p.name not in allowed]
    if unexpected:
        raise OutputDirectoryConflict('Use a new empty output directory; existing files: '+', '.join(sorted(unexpected)[:8]))

def read_repeat_stack(path):
    with tifffile.TiffFile(path) as source:
        if len(source.series)!=1:raise ValueError('Use one unambiguous single-channel repeat stack')
        series=source.series[0]
        if series.axes not in {'TYX','IYX','QYX','FYX'}:
            raise ValueError('Declared TIFF axes must be repeat/Y/X, not Z, channel, or RGB: '+series.axes)
        if any(page.samplesperpixel!=1 for page in series.pages):
            raise ValueError('RGB/multisample TIFF is not a grayscale repeat stack')
        stack=np.asarray(series.asarray())
    if stack.ndim!=3 or stack.dtype not in (np.dtype('uint16'),np.dtype('float32')):
        raise ValueError('Expected FYX uint16 or float32 unmodified repeat stack')
    return stack

def check_stop(out):
    if (out/'STOP_REQUESTED.json').exists():raise Cancelled('Cancellation requested; no further model call started.')

def progress(out,stage,done=0,total=1):
    check_stop(out)
    write_json(out/'progress.json',dict(stage=stage,done=done,total=total,time=time.time()))

def write_plane(path,pixels,request):
    calibration=request.get('pixel_calibration',{})
    pw,ph=calibration.get('pixel_width',1.),calibration.get('pixel_height',1.)
    if not np.isfinite([pw,ph]).all() or min(pw,ph)<=0:raise ValueError('Invalid spatial pixel calibration')
    options=dict(resolution=(1./pw,1./ph),resolutionunit='NONE',
        metadata=dict(axes='YXS' if pixels.ndim==3 else 'YX',pixel_calibration=calibration))
    if pixels.ndim==3:
        options['photometric']='rgb'
        if pixels.shape[-1]==4:options['extrasamples']=['UNASSALPHA']
    tifffile.imwrite(path,pixels,**options)

def run_loop(query,predict,report,bank,out,request=None):
    Y=np.empty((K,T,*query.shape),np.float32)
    R=np.empty((K,T-1,*query.shape),np.float32)
    progress(out,'single_pass_denoising',0,11)
    Y[:,0]=predict(query)
    write_plane(out/'D1.tif',Y[0,0],request or {})
    calls=1;streams=[]
    for k in range(K):
        for t in range(T-1):
            progress(out,'calibrated_loop',calls,11)
            seed=[SEED,k,t]
            R[k,t]=sample_noise(Y[k,t],np.random.default_rng(np.random.SeedSequence(seed)),report,bank)
            Y[k,t+1]=predict(R[k,t])
            calls+=1;streams.append(seed)
    if not np.isfinite(Y).all() or not np.isfinite(R).all():raise FloatingPointError('Nonfinite trajectory')
    return Y,R,dict(D_calls=calls,R_calls=10,K=K,T=T,seeds=streams,D1_reused=True)

def map_and_checks(Y,sigma,chart):
    means=Y.astype(float).mean(axis=(-2,-1),keepdims=True)
    centered=(Y.astype(float)-means)/chart['scale']
    mapped=signature_map_from_Y(centered,horizon=T,eps=sigma/chart['scale'],grad_sigma=2.,
        map_version='horizon_v1',ratio_eps=1e-6,structure_floor=1e-6)
    rgb=render_2d_colormap(mapped['gamma'],mapped['C_stab'],
        structure_valid=mapped['structure_valid'],gamma_log_limits=(-.4,.4),cstab_limits=(0,1))
    sig=signature_from_Y((Y.astype(float)-chart['offset'])/chart['scale'],
        gain_mode='global_mean',gain_contrast_norm='none')
    sig.update(T=T,K=K)
    checks=verdict(sig)
    candidate,reason,_=decide_from_energy(checks)
    return mapped,rgb,dict(signature=sig,checks=checks,compatibility_readout=candidate,
        compatibility_reason=reason,profile='LEGACY_CAPS_DIAGNOSTIC_ONLY',
        deployment_qualified=False,photometry='fixed model chart; scalar gate not DC-projected',
        limits=dict(near_inv_cap=.002,var_cap=.00005,gain_band_delta=.10,diss_tol=1.10)),means

def overview(out,query,Y,R,rgb,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    lo,hi=np.percentile(query,[1,99.5])
    for ax,im,title in zip(axes[0],[query,*Y[0]],['Input','D1 (single pass)','D2','D3']):
        ax.imshow(im,cmap='gray',vmin=lo,vmax=hi);ax.set_title(title);ax.axis('off')
    for ax,im,title in zip(axes[1,:2],R[0],['R1 (empirical measured challenge)','R2']):
        ax.imshow(im,cmap='gray',vmin=lo,vmax=hi);ax.set_title(title);ax.axis('off')
    axes[1,2].imshow(rgb);axes[1,2].set_title('Response overlay (1 px support margin)');axes[1,2].axis('off')
    g,c=np.meshgrid(np.exp(np.linspace(-.4,.4,200)),np.linspace(0,1,160))
    legend=render_2d_colormap(g,c,gamma_log_limits=(-.4,.4),cstab_limits=(0,1))
    axes[1,3].imshow(legend,origin='lower',extent=(-.4,.4,0,1),aspect='auto')
    axes[1,3].set_xlabel('log structure gain (fixed range)');axes[1,3].set_ylabel('Stability score (not probability)')
    axes[1,3].set_title('Numerical two-axis legend')
    fig.suptitle('Measured same-scene perturbation | not a calibrated error map or deployment certificate\n'
        f"Noise RMS = {report['measured_noise_sigma_native']:.4g} native units; common image display range [{lo:.4g}, {hi:.4g}]")
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'overview.png',dpi=140);plt.close(fig)

def render_report(out,response,report):
    cards=''
    for title,key in [('Noise qualification','calibration_json'),('Manifest','manifest_json'),
        ('D1 floating-point TIFF','d1_tiff'),('Structure-aware Map overlay TIFF','map_rgb_tiff'),
        ('Original unmasked Map TIFF','map_unmasked_rgb_tiff'),('Transparent Map layer TIFF','map_rgba_tiff'),
        ('Display mask (not a validity mask)','map_display_mask_tiff'),
        ('Display protocol and limitations','map_display_json'),
        ('Numerical two-axis legend','map_legend_png'),('D trajectory (K-major, T-minor)','trajectory_tiff'),
        ('R trajectory (K-major, T-minus-one-minor)','renoise_tiff'),('Stability coordinate TIFF','stability_tiff'),
        ('Structure-gain coordinate TIFF','gain_tiff'),('All numerical Map components','components_npz'),
        ('P1-P4 diagnostic CSV','diagnostics_csv')]:
        if response['artifacts'].get(key):
            cards+=f'<li><a href="{html.escape(Path(response["artifacts"][key]).name)}">{title}</a></li>'
    preview='<img src="overview.png" alt="Loop and Map overview">' if (out/'overview.png').exists() else ''
    page='''<!doctype html><meta charset="utf-8"><title>LoopSN2N calibrated workflow</title>
<style>body{max-width:1250px;margin:35px auto;padding:0 25px;font:16px/1.6 system-ui;background:#f5f7fa;color:#162b3b}section{background:white;padding:24px;border-radius:12px;margin:18px 0}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{font-size:28px}strong{color:#a45c16}</style>
<h1>LoopSN2N | Noise-calibrated research workflow</h1>'''
    page+=f'<section><p>Execution: <b>{html.escape(response["execution_status"])}</b></p><p>Noise qualification: <b>{html.escape(response["calibration_status"])}</b></p><p>Deployment decision: <strong>{html.escape(response["decision"]["label"])}</strong></p><p>{html.escape(response["decision"]["reason"])}</p><ul>'+''.join('<li>'+html.escape(w)+'</li>' for w in response['warnings'])+'</ul></section>'
    page+='<section>'+preview+'<ul>'+cards+'</ul></section><section><h2>Calibration evidence</h2><pre>'+html.escape(json.dumps(report,indent=2,ensure_ascii=False,default=serial))+'</pre></section>'
    (out/'report.html').write_text(page,encoding='utf-8')

def _evaluate(request,*,predictor_factory=None):
    if request.get('schema_version')!=1:raise ValueError('schema_version must be 1')
    if request.get('domain')!='static_repeats':raise ValueError('Only explicitly declared static_repeats are supported; phase/Z/live frames are not repeated measurements.')
    trust = request.get('trust_checkpoint', False)
    if not isinstance(trust, bool):raise ValueError('trust_checkpoint must be a boolean')
    out=Path(request['output_dir']).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Refuse to overwrite a completed run')
    validate_output_directory(out)
    progress(out,'read_stack')
    stack_path=Path(request['stack_path']).resolve(strict=True)
    checkpoint=Path(request['checkpoint_path']).resolve(strict=True)
    stack=read_repeat_stack(stack_path)
    if stack.ndim!=3:raise ValueError('Expected unambiguous FYX single-channel stack')
    qi=request.get('query_index',0)
    if isinstance(qi,bool) or not isinstance(qi,int) or not 0<=qi<len(stack):raise ValueError('query_index out of range')
    progress(out,'calibrate_noise')
    report,bank=calibrate_stack(stack,query_index=qi,domain=request['domain'],output_dir=out)
    write_json(out/'calibration.json',report)
    query=stack[qi].astype(np.float32)
    chart=request.get('chart')
    if chart is not None and (not isinstance(chart,dict) or not {'offset','scale'} <= set(chart) or not np.isfinite([chart['offset'],chart['scale']]).all() or chart['scale']<=0):
        raise ValueError('Finite offset and positive fixed scale are required')
    response=dict(schema_version=1,status='COMPLETE',execution_status='COMPLETED_WITH_ABSTENTION',
        calibration_status=report['qualification']['status'],
        decision=dict(label='DEFER',reason='This new empirical challenge and fixed inference profile have no validated deployment threshold profile. Compatibility P1-P4 are diagnostic only.',diagnostic_checks={}),
        warnings=['This is a same-scene empirical challenge, not a physical camera-noise certificate.',
            'Static repeated frames are a user-declared assumption; motion/phase/Z stacks are not interchangeable.',
            'The Loop Response Map is a spatial response display, not pixel error or a probability.',
            'Only the selected full plane is mapped; the stack supplies calibration repeats.',
            'Blue/orange structure overlay uses a 1-pixel support margin; grayscale retains D1, not zero response. The unmasked Map and numerical components are also saved.'],artifacts={})
    artifacts=response['artifacts']
    artifacts.update(calibration_json=str(out/'calibration.json'),manifest_json=str(out/'manifest.json'),report_html=str(out/'report.html'))
    manifest=dict(schema=SCHEMA,request=request,stack_sha256=sha(stack_path),checkpoint_sha256=sha(checkpoint),
        input_shape=list(stack.shape),input_dtype=str(stack.dtype),query_index=qi,chart=chart,
        chart_source='EXPLICIT_REQUEST' if chart is not None else 'NOT_YET_REQUIRED',
        calibration=report,started=time.time(),python=sys.version,platform=platform.platform(),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'fiji_calibrated_plugin/backend').rglob('*.py'))})
    tifffile.imwrite(out/'input_plane.tif',query,metadata={'axes':'YX'})
    if response['calibration_status']!='EMPIRICAL_SAME_SCENE_QUALIFIED':
        response['decision']['reason']='Noise qualification failed; no unqualified noise was injected and no Map was fabricated. See calibration reasons.'
        response['warnings'].append('Loop was not run. The completed workflow intentionally abstained at the calibration check.')
        manifest['loop_executed']=False
    else:
        from fiji_calibrated_plugin.backend.model_adapter import MissingChartError
        if predictor_factory is None:
            from fiji_calibrated_plugin.backend.model_adapter import load_predictor
            predictor_factory=load_predictor
        progress(out,'load_checkpoint')
        try:
            model=predictor_factory(str(checkpoint),device=request.get('device','auto'),trust_checkpoint=trust,chart=request.get('chart'))
        except MissingChartError:
            chart=dict(offset=float(query.min()),scale=float(query.max()-query.min()))
            if not np.isfinite([chart['offset'],chart['scale']]).all() or chart['scale']<=0:
                raise ValueError('Checkpoint has no fixed chart and query range is degenerate. Supply the checkpoint training chart explicitly.')
            model=predictor_factory(str(checkpoint),device=request.get('device','auto'),trust_checkpoint=trust,chart=chart)
            response['warnings'].append('Checkpoint has no recorded training chart: query min/max was frozen once for exploratory inference, not training-domain qualification.')
        chart=model.manifest['chart']
        manifest['chart']=chart
        manifest['chart_source']=model.manifest.get('chart_source','ADAPTER_DECLARED')
        manifest['model']=model.manifest
        Y,R,calls=run_loop(query,model.predict,report,bank,out,request)
        model.assert_unchanged()
        progress(out,'full_plane_map',11,11)
        mapped,rgb,diagnostics,means=map_and_checks(Y,report['measured_noise_sigma_native'],chart)
        response['decision']['diagnostic_checks']=diagnostics['checks']
        response['diagnostics']=diagnostics
        manifest.update(loop_executed=True,calls=calls,diagnostics=diagnostics,map=dict(
            profile='E69_horizon_v1_fixed_model_units_empirical_epsilon_v1',epsilon_native=report['measured_noise_sigma_native'],
            epsilon_model=report['measured_noise_sigma_native']/chart['scale'],numeric_units='fixed_model_chart_dimensionless',
            epsilon_source='FIT_PAIR_BANK_NOT_DENOISER_OUTPUT',ratio_eps=1e-6,structure_floor=1e-6,
            stability_denominator_floor=1e-12,grad_sigma_pixels=2.,gamma_log_limits=[-.4,.4],
            cstab_limits=[0,1],DC_projection='(Y_native-full_plane_mean_native)/fixed_chart_scale; map only',full_plane_coverage=True))
        np.save(out/'D_native.npy',Y);np.save(out/'R_native.npy',R)
        np.savez_compressed(out/'map_components.npz',**{n:v for n,v in mapped.items() if isinstance(v,np.ndarray)},image_means=means)
        write_plane(out/'D1.tif',Y[0,0],request)
        tifffile.imwrite(out/'D_trajectory.tif',Y.reshape(K*T,*query.shape),metadata={'axes':'IYX'})
        tifffile.imwrite(out/'R_trajectory.tif',R.reshape(K*(T-1),*query.shape),metadata={'axes':'IYX'})
        original_rgb=np.rint(rgb*255).astype(np.uint8)
        write_plane(out/'map_rgb_unmasked.tif',original_rgb,request)
        progress(out,'structure_aware_display',11,11)
        display=render_support_display(query,Y,report['measured_noise_sigma_native'],
                                       chart,original_rgb,check_stop=lambda:check_stop(out))
        write_plane(out/'map_rgb.tif',display['rgb'],request)
        write_plane(out/'map_rgba.tif',display['rgba'],request)
        write_plane(out/'D1_display_rgb.tif',display['d1_gray_rgb'],request)
        for key in ('support','core','raw_core','loop_core'):
            write_plane(out/('display_'+key+'.tif'),display[key].astype(np.uint8),request)
        write_json(out/'display_protocol.json',display['metadata'])
        manifest['display']=display['metadata']
        from PIL import Image
        Image.fromarray(display['rgb']).save(out/'map_overlay.png')
        Image.fromarray(display['rgba']).save(out/'map_rgba.png')
        Image.fromarray(original_rgb).save(out/'map_unmasked.png')
        # A numerical two-axis legend, not a univariate scalar colour bar.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        g,c=np.meshgrid(np.exp(np.linspace(-.4,.4,200)),np.linspace(0,1,160))
        legend=render_2d_colormap(g,c,gamma_log_limits=(-.4,.4),cstab_limits=(0,1))
        fig,ax=plt.subplots(figsize=(5.6,3.8))
        ax.imshow(legend,origin='lower',extent=(-.4,.4,0,1),aspect='auto')
        ax.set_xlabel('log structure gain: blue attenuation / orange enhancement')
        ax.set_ylabel('Stability score (not probability)')
        ax.set_title('Original bivariate colour scale')
        fig.tight_layout();fig.savefig(out/'map_legend.png',dpi=160);plt.close(fig)
        for key in ('C_stab','gamma'):
            tifffile.imwrite(out/(key+'.tif'),mapped[key].astype(np.float32),metadata={'axes':'YX'})
        with (out/'diagnostics.csv').open('x',newline='',encoding='utf-8') as file:
            writer=csv.writer(file);writer.writerow(['section','name','value'])
            writer.writerow(['deployment','label','DEFER'])
            for section in ('signature','checks'):
                for name,value in diagnostics[section].items():
                    writer.writerow([section,name,json.dumps(value,default=serial,allow_nan=False)])
            writer.writerow(['diagnostic_only','compatibility_readout',diagnostics['compatibility_readout']])
        overview(out,query,Y,R,display['rgb'],report)
        artifacts.update(d1_tiff=str(out/'D1.tif'),map_rgb_tiff=str(out/'map_rgb.tif'),
            map_unmasked_rgb_tiff=str(out/'map_rgb_unmasked.tif'),map_rgba_tiff=str(out/'map_rgba.tif'),
            map_display_mask_tiff=str(out/'display_support.tif'),map_display_json=str(out/'display_protocol.json'),
            map_legend_png=str(out/'map_legend.png'),overview_png=str(out/'overview.png'),trajectory_tiff=str(out/'D_trajectory.tif'),
            renoise_tiff=str(out/'R_trajectory.tif'),stability_tiff=str(out/'C_stab.tif'),gain_tiff=str(out/'gamma.tif'),
            components_npz=str(out/'map_components.npz'),diagnostics_csv=str(out/'diagnostics.csv'))
    manifest['seconds']=time.time()-manifest['started'];manifest['result']=response
    check_stop(out)
    render_report(out,response,report)
    write_json(out/'progress.json',dict(stage='COMPLETE',done=1,total=1))
    write_json(out/'manifest.json',manifest)
    return response

def evaluate(request,*,predictor_factory=None):
    # A local atomic lock protects one job across independent Java/CLI callers.
    out=Path(request['output_dir']).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Refuse to overwrite a completed run')
    lock=out/'RUNNING.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    try:
        validate_output_directory(out)
        return _evaluate(request,predictor_factory=predictor_factory)
    finally:lock.unlink()

def failure_response(request,exc,*,owns_lock=False):
    out=Path(request['output_dir']).resolve();out.mkdir(parents=True,exist_ok=True)
    report_path=out/'calibration.json'
    report=json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    progress_path=out/'progress.json'
    last=json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else {}
    response=dict(schema_version=1,status='CANCELLED' if isinstance(exc,Cancelled) else 'FAILED',
        execution_status='CANCELLED' if isinstance(exc,Cancelled) else 'FAILED',
        calibration_status=report.get('qualification',{}).get('status','NOT_COMPLETED'),
        decision=dict(label='DEFER',reason=str(exc),diagnostic_checks={}),
        warnings=['Incomplete run: no deployment decision is available. Existing measurements are retained.'],
        artifacts={},error=dict(code=type(exc).__name__,message=str(exc),stage=last.get('stage','preflight')))
    for name,file in [('calibration_json','calibration.json'),('d1_tiff','D1.tif')]:
        if (out/file).exists():response['artifacts'][name]=str(out/file)
    # Do not alter an existing result or an active owner's manifest on refusal.
    if owns_lock or (not (out/'manifest.json').exists() and not (out/'RUNNING.lock').exists()):
        response['artifacts'].update(manifest_json=str(out/'manifest.json'),report_html=str(out/'report.html'))
        render_report(out,response,report)
        write_json(out/'progress.json',dict(stage=response['status'],done=0,total=1))
        write_json(out/'manifest.json',dict(schema=SCHEMA,request=request,result=response,last_progress=last,complete=False))
    return response

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--response',required=True)
    args=p.parse_args();response_path=Path(args.response).resolve()
    request=json.loads(Path(args.request).read_text(encoding='utf-8-sig'))
    out=Path(request['output_dir']).resolve();out.mkdir(parents=True,exist_ok=True)
    if response_path!=out/'response.json':raise ValueError('Response must be the job response.json')
    lock=out/'RUNNING.lock'
    # A caller without ownership must never publish a response for this job.
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    try:
        validate_output_directory(out)
        if response_path.exists() or (out/'manifest.json').exists():raise FileExistsError('Refuse to overwrite completed response/manifest')
        try:
            result=_evaluate(request)
        except OutputDirectoryConflict:
            raise  # Preserve every file in a reused directory, including inputs.
        except Exception as exc:
            result=failure_response(request,exc,owns_lock=True)
            traceback.print_exc()
        write_json(response_path,result)
    finally:lock.unlink()
    if result['status']=='FAILED':sys.exit(1)
if __name__=='__main__':main()
