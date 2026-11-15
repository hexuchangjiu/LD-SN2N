"""Measured-instance raw camera, BioSR MT Cell_002 L05; no reconstruction-domain perturbation.
Runtime uses numpy only. Model inputs are float32 states in the original denoiser's
chart. Exposure is synthesized anew from the CURRENT state, an optical carrier,
a measured OTF and fitted raw radiometry. This is not instrument calibration.
"""
from pathlib import Path
import numpy as np

def project(state, carrier, otf):
    s=np.asarray(state,dtype=np.float32)
    if s.shape!=(1004,1004) or not np.isfinite(s).all():
        raise ValueError("registered MT state must be finite (1004,1004)")
    # fairSIM's native pixel j is registered on the supersampled image at 2*j,
    # whereas the detector integral over samples (2*j,2*j+1) has centroid 2*j+0.5.
    # Apply the fixed coordinate conversion before optical modulation, not by
    # registering/restretching output images. Exactly the Fourier translation
    # tested by the independent native-coordinate phantom (0.5 HR px per axis).
    n=s.shape[0]; phase=np.exp(-1j*np.pi*(np.fft.fftfreq(n)[:,None]+np.fft.fftfreq(n)[None,:]))
    s=np.fft.ifft2(np.fft.fft2(s)*phase).real.astype('<f4')
    # Clamp only the emitter proxy to nonnegative, not each normalized observation.
    s=np.maximum(s,0)
    def mode(field):
        # Real radial OTF is stored on native high-resolution rFFT grid. Integrate 2x2 detector bins.
        y=np.fft.irfft2(np.fft.rfft2(field)*otf,s=s.shape)
        return y.reshape(502,2,502,2).mean((1,3)).astype('f4')
    return mode(s),np.stack([mode(s*c) for c in carrier])

class Camera:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as f:
            self.carrier=f['carrier']; self.otf=f['otf']; self.gain_dc=f['gain_dc']; self.gain_ac=f['gain_ac']
            self.bg=f['bg']; self.photon_per_adu=float(f['photon_per_adu'])
            self.bias=float(f['bias']); self.read_sd=float(f['read_sd'])
    def expected(self,state):
        dc,ac=project(state,self.carrier,self.otf)
        excess=self.bg+self.gain_dc[:,None,None]*dc+self.gain_ac[:,None,None]*ac
        if not np.isfinite(excess).all(): raise ValueError("nonfinite phase mean")
        return np.maximum(excess,0)*self.photon_per_adu
    def sample(self,mu,rng):
        mu=np.asarray(mu)
        if mu.shape!=(9,502,502) or not np.isfinite(mu).all() or mu.min()<0: raise ValueError("invalid raw expected photons")
        # Poisson draws in photon coordinates; convert back to the stored count
        # domain before adding read/residual term (ADU), camera bias and quantization.
        y=rng.poisson(mu)/self.photon_per_adu+rng.normal(0.,self.read_sd,mu.shape)+self.bias
        return np.clip(np.rint(y),0,65535).astype('<f4')
def load_camera(asset_dir):
    return Camera(Path(asset_dir)/'camera_model.npz')

