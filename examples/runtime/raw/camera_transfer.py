"""Transfer the original empirical camera without refitting a single parameter.
Use original fixed spatial regions and original acceptance gates. No proxy reference.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
def qualify(cq,cam,state,measured):
    Y,X=np.mgrid[:502,:502]; hold=((X//48+2*(Y//48))%4==0)
    te=hold&(X>=12)&(X<490)&(Y>=12)&(Y<490)
    mask=~hold
    div=gaussian_filter(mask.astype('f8'),30.,mode='reflect')
    mu=cam.expected(state)
    pred=mu/cam.photon_per_adu+cam.bias
    rows=[]
    for y,p in zip(measured,pred):
        back=gaussian_filter(np.where(mask,y-cam.bias,0.).astype('f8'),30.,mode='reflect')/div+cam.bias
        rows.append(dict(mse=float(np.mean((p[te]-y[te])**2)),background_mse=float(np.mean((back[te]-y[te])**2)),pearson=float(np.corrcoef(p[te],y[te])[0,1]),mean_relative_bias=float((p[te]-y[te]).mean()/(y[te].mean()-cam.bias))))
    g=dict(cq['gates']); errs=sum(v['mse'] for v in rows)/sum(v['background_mse'] for v in rows)
    residual=[]
    for seed in [417111,417112]:
        s=cam.sample(mu,np.random.default_rng(seed))
        residual.append((s-pred).ravel())
    g.update(mean=bool(max(abs(v['mean_relative_bias']) for v in rows)<.10),per_frame_corr=bool(min(v['pearson'] for v in rows)>.50),relative_mse=bool(errs<.80),finite_means=bool(np.isfinite(mu).all()),shape=bool(mu.shape==(9,502,502)),state_dependence=bool(np.mean(abs(cam.expected(2*state)-mu))>.01 and np.mean(abs(mu-cam.expected(state*0)))>.01),independent_draws=bool(abs(np.corrcoef(*residual)[0,1])<.01))
    return dict(gates=g,status='PASS' if all(g.values()) else 'FAILED',mse_ratio=errs,phase_rows=rows,noise_fit_reused=True,refit=False,scope='Same measured source, same measured D1 input, same spatial tiles; baseline camera frozen in original D1 chart. Same-source technical transfer not external validation.')
