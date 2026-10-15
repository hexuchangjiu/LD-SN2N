"""Synthetic contract tests; no claim of microscopy efficacy."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import tifffile
from fiji_calibrated_plugin.backend import bridge as b

class FixedFake:
    def __init__(self,*args,chart=None,**kwargs):
        self.manifest={'chart':chart or {'offset':0.,'scale':255.},'chart_source':'TEST_ONLY'}
    def predict(self,x):return np.asarray(x,dtype=np.float32)*.98
    def assert_unchanged(self):pass

class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        rng=np.random.default_rng(6901);yy,xx=np.mgrid[:32,:40]
        scene=100+20*np.sin(xx/5)+10*np.cos(yy/7)
        self.stack=scene+rng.normal(0,3,(80,32,40))
        tifffile.imwrite(self.root/'input.tif',self.stack.astype(np.float32),metadata={'axes':'TYX'})
        (self.root/'dummy.pt').write_bytes(b'TEST_ONLY_NOT_WEIGHTS')
        self.request=dict(schema_version=1,domain='static_repeats',trust_checkpoint=True,
            stack_path=str(self.root/'input.tif'),checkpoint_path=str(self.root/'dummy.pt'),
            output_dir=str(self.root/'run'),query_index=0,device='cpu')
    def tearDown(self):self.temp.cleanup()
    def test_complete_preserves_dimensions_and_abstains(self):
        out=b.evaluate(self.request,predictor_factory=FixedFake)
        self.assertEqual(out['status'],'COMPLETE');self.assertEqual(out['decision']['label'],'DEFER')
        self.assertEqual(out['calibration_status'],'EMPIRICAL_SAME_SCENE_QUALIFIED')
        self.assertEqual(tifffile.imread(out['artifacts']['map_rgb_tiff']).shape,(32,40,3))
        y=np.load(self.root/'run/D_native.npy');r=np.load(self.root/'run/R_native.npy')
        self.assertEqual(y.shape,(5,3,32,40));self.assertEqual(r.shape,(5,2,32,40))
        self.assertTrue(np.array_equal(y[0,0],y[4,0]))
        m=json.loads((self.root/'run/manifest.json').read_text())
        self.assertEqual(m['calls']['D_calls'],11);self.assertTrue(m['source_sha256'])
        self.assertIn('not pixel error or a probability',(self.root/'run/report.html').read_text())
        with self.assertRaises(FileExistsError):b.evaluate(self.request,predictor_factory=FixedFake)
    def test_invalid_domain_rejected_before_loading(self):
        for domain in ['z_stack','sim_phases','live','unknown']:
            with self.assertRaises(ValueError):b.evaluate(dict(self.request,domain=domain),predictor_factory=FixedFake)

    def test_explicit_z_rgb_and_8bit_rejected(self):
        for axes,pixels,kwargs in [
            ('ZYX',self.stack.astype(np.float32),{}),
            ('YXS',np.ones((80,32,3),np.uint16),{'photometric':'rgb'}),
            ('TYX',self.stack.astype(np.uint8),{})]:
            tifffile.imwrite(self.root/'bad.tif',pixels,metadata={'axes':axes},**kwargs)
            with self.assertRaises(ValueError):b.read_repeat_stack(self.root/'bad.tif')

    def test_cli_reused_directory_preserves_all_existing_files(self):
        import subprocess,sys
        out=Path(self.request['output_dir']);out.mkdir()
        prior=b'original user image'
        (out/'input_plane.tif').write_bytes(prior)
        b.write_json(out/'request.json',self.request)
        result=subprocess.run([sys.executable,str(Path(b.__file__)), '--request',str(out/'request.json'),
            '--response',str(out/'response.json')],capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual((out/'input_plane.tif').read_bytes(),prior)
        self.assertFalse((out/'response.json').exists())
        self.assertFalse((out/'manifest.json').exists())

    def test_malformed_cli_cannot_overwrite_prior_error_report(self):
        import subprocess,sys
        out=Path(self.request['output_dir']);out.mkdir()
        (out/'report.html').write_text('KEEP PRIOR REPORT')
        b.write_json(out/'request.json',dict(self.request,domain='invalid'))
        result=subprocess.run([sys.executable,str(Path(b.__file__)), '--request',str(out/'request.json'),
            '--response',str(out/'response.json')],capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual((out/'report.html').read_text(),'KEEP PRIOR REPORT')
        self.assertFalse((out/'response.json').exists())
    def test_unqualified_does_not_infer_or_create_map(self):
        altered=self.stack+np.arange(80)[:,None,None]*5
        tifffile.imwrite(self.root/'input.tif',altered.astype(np.float32),metadata={'axes':'TYX'})
        def forbidden(*a,**kw):raise AssertionError('Must not load model')
        out=b.evaluate(self.request,predictor_factory=forbidden)
        self.assertEqual(out['calibration_status'],'NOT_QUALIFIED')
        self.assertEqual(out['decision']['label'],'DEFER')
        self.assertNotIn('map_rgb_tiff',out['artifacts'])
        self.assertTrue((self.root/'run/report.html').is_file())
    def test_stop_before_read(self):
        dest=Path(self.request['output_dir']);dest.mkdir();(dest/'STOP_REQUESTED.json').write_text('{}')
        with self.assertRaises(b.Cancelled):b.evaluate(self.request,predictor_factory=FixedFake)
    def test_noise_sigma_not_derived_from_denoiser(self):
        from fiji_calibrated_plugin.backend.noise_calibration import calibrate_stack
        report,bank=calibrate_stack(self.stack)
        self.assertAlmostEqual(report['measured_noise_sigma_native'],3,delta=.15)
        q=self.stack[0].astype(np.float32)
        dest=self.root/'calls';dest.mkdir()
        y,r,calls=b.run_loop(q,lambda x:x,report,bank,dest)
        y2,r2,_=b.run_loop(q,lambda x:x,report,bank,dest)
        self.assertTrue(np.array_equal(y,y2));self.assertTrue(np.array_equal(r,r2))
    def test_frozen_map_identity_no_formula_rewrite(self):
        import ast,hashlib
        root=Path(__file__).resolve().parents[1]
        # Frozen release-source AST digests; independent of a private checkout.
        expected={'energy.py':'b216821ff09be55e1ca6bf2a8ab39784d301e8658b8515d3f54a419e265ee607',
                  'loop_map.py':'13fa299f412783f02b812a928f2139029c585431ecc586eafe333422ce212d70'}
        for name,digest in expected.items():
            node=ast.dump(ast.parse((root/'fiji_calibrated_plugin/backend/vendor'/name).read_text(encoding='utf-8-sig')))
            self.assertEqual(hashlib.sha256(node.encode()).hexdigest(),digest)

    def test_weights_only_request_does_not_enable_legacy_trust(self):
        seen=[]
        def factory(*args,**kw):
            seen.append(kw['trust_checkpoint'])
            return FixedFake(*args,**kw)
        b.evaluate(dict(self.request,trust_checkpoint=False),predictor_factory=factory)
        self.assertEqual(seen,[False])

    def test_reused_output_preserves_existing_pixels(self):
        out=Path(self.request['output_dir']);out.mkdir()
        prior=b'prior user output must survive'
        (out/'D1.tif').write_bytes(prior)
        with self.assertRaises(b.OutputDirectoryConflict):b.evaluate(self.request,predictor_factory=FixedFake)
        self.assertEqual((out/'D1.tif').read_bytes(),prior)
        self.assertFalse((out/'manifest.json').exists())
        self.assertFalse((out/'RUNNING.lock').exists())

    def test_constant_query_can_use_checkpoint_chart(self):
        altered=self.stack.copy();altered[:2]=100
        tifffile.imwrite(self.root/'input.tif',altered.astype(np.float32),metadata={'axes':'TYX'})
        result=b.evaluate(self.request,predictor_factory=FixedFake)
        self.assertEqual(result['status'],'COMPLETE')
        m=json.loads((self.root/'run/manifest.json').read_text())
        self.assertEqual(m['chart_source'],'TEST_ONLY')
    def test_map_invariant_to_native_unit_and_offset_change(self):
        rng=np.random.default_rng(12)
        y=rng.normal(100,10,(5,3,24,32));y[:,0]=y[0,0]
        a,_,_,_=b.map_and_checks(y,3.,{'offset':50.,'scale':255.})
        c,_,_,_=b.map_and_checks((y+200)/65535,3/65535,{'offset':250/65535,'scale':255/65535})
        for key in ['gamma','C_stab','sigma2_K','sigma2_T']:
            np.testing.assert_allclose(a[key],c[key],rtol=1e-10,atol=1e-12)
    def test_failure_report_keeps_completed_calibration(self):
        out=Path(self.request['output_dir']);out.mkdir()
        b.write_json(out/'calibration.json',{'qualification':{'status':'EMPIRICAL_SAME_SCENE_QUALIFIED'}})
        b.write_json(out/'progress.json',{'stage':'calibrated_loop'})
        tifffile.imwrite(out/'D1.tif',np.ones((10,10),np.float32))
        result=b.failure_response(self.request,ValueError('Signal out of range'))
        self.assertEqual(result['calibration_status'],'EMPIRICAL_SAME_SCENE_QUALIFIED')
        self.assertIn('calibration_json',result['artifacts']);self.assertIn('d1_tiff',result['artifacts'])
        self.assertTrue((out/'report.html').exists());self.assertEqual(result['decision']['label'],'DEFER')
    def test_spatial_calibration_export(self):
        p=self.root/'spatial.tif'
        b.write_plane(p,np.ones((5,7),np.float32),{'pixel_calibration':{'pixel_width':.04,'pixel_height':.05,'unit':'um'}})
        with tifffile.TiffFile(p) as tf:
            x=tf.pages[0].tags['XResolution'].value;y=tf.pages[0].tags['YResolution'].value
            self.assertAlmostEqual(x[0]/x[1],25);self.assertAlmostEqual(y[0]/y[1],20)
            self.assertEqual(json.loads(tf.pages[0].description)['pixel_calibration']['unit'],'um')
    def test_busy_cli_cannot_write_owners_response(self):
        import subprocess,sys
        out=Path(self.request['output_dir']);out.mkdir()
        (out/'RUNNING.lock').write_text('another-owner')
        b.write_json(out/'request.json',self.request)
        result=subprocess.run([sys.executable,str(Path(b.__file__)), '--request',str(out/'request.json'),
            '--response',str(out/'response.json')],capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertFalse((out/'response.json').exists());self.assertFalse((out/'manifest.json').exists())
        self.assertEqual((out/'RUNNING.lock').read_text(),'another-owner')
if __name__=='__main__':unittest.main()
