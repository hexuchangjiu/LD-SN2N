"""Display fidelity and contrast guards, not segmentation efficacy tests."""
import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt
from fiji_calibrated_plugin.backend.support_display import detect_support, render_support_display


class SupportDisplayTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.mgrid[:96, :112]
        self.raw = 10 + 5 * np.exp(-.5*((xx-40)/2.)**2)
        self.y = np.stack([self.raw*.98]*3)[None].repeat(5, axis=0)
        self.rgb = np.random.default_rng(45).integers(0, 256, (96,112,3), dtype=np.uint8)
        self.chart = {'offset':0., 'scale':20.}

    def render(self, **kw):
        return render_support_display(kw.get('raw',self.raw), kw.get('y',self.y),
            kw.get('sigma',1.), kw.get('chart',self.chart), self.rgb)

    def test_frozen_prototype_detector_parity(self):
        import ast,hashlib
        from pathlib import Path
        from fiji_calibrated_plugin.backend import support_display
        node=ast.dump(ast.parse(Path(support_display.__file__).read_text(encoding='utf-8-sig')))
        self.assertEqual(hashlib.sha256(node.encode()).hexdigest(),
            'f87b3697260a5994ffc319ff1e9d78bc723fa059be5fd56eae442abe167cce0b')

    def test_one_pixel_margin_and_original_colour(self):
        a=self.render()
        np.testing.assert_array_equal(a['support'],distance_transform_edt(~a['core'])<=1)
        np.testing.assert_array_equal(a['rgb'][a['support']],self.rgb[a['support']])
        np.testing.assert_array_equal(a['rgb'][~a['support']],a['d1_gray_rgb'][~a['support']])
        np.testing.assert_array_equal(a['rgba'][...,:3],self.rgb)
        self.assertEqual(set(np.unique(a['rgba'][...,3])),{0,255})

    def test_brightness_units_invariant(self):
        a=self.render()
        b=self.render(raw=self.raw*4+100,y=self.y*4+100,sigma=4.,chart={'offset':100.,'scale':80.})
        for key in ('core','support','rgb','rgba'):
            np.testing.assert_array_equal(a[key],b[key])

    def test_protect_raw_structure_even_if_D1_erases_it(self):
        a=self.render(y=np.ones_like(self.y)*10)
        self.assertTrue(a['raw_core'][:,40].all())
        self.assertTrue(a['support'][:,40].all())

    def test_protect_Loop_only_core(self):
        y=self.y.copy()
        y[:,0]=10
        a=self.render(raw=np.ones_like(self.raw)*10,y=y)
        self.assertFalse(a['raw_core'].any())
        self.assertTrue(a['loop_core'][:,40].all())
        self.assertTrue(a['support'][:,40].all())

    def test_empty_constant_support_still_retains_D1(self):
        a=self.render(raw=np.ones_like(self.raw)*10,y=np.ones_like(self.y)*10)
        self.assertFalse(a['support'].any())
        np.testing.assert_array_equal(a['rgb'],a['d1_gray_rgb'])

    def test_nonfinite_and_different_initial_states_rejected(self):
        y=self.y.copy();y[3,0,0,0]+=1
        with self.assertRaises(ValueError):self.render(y=y)
        y=self.y.copy();y[2,2,0,0]=np.nan
        with self.assertRaises(ValueError):self.render(y=y)
        for sigma in (0.,float('nan'),-1.):
            with self.assertRaises(ValueError):self.render(sigma=sigma)

    def test_inputs_unchanged_and_stop_callback(self):
        q,y,rgb=self.raw.copy(),self.y.copy(),self.rgb.copy()
        self.render()
        np.testing.assert_array_equal(q,self.raw);np.testing.assert_array_equal(y,self.y)
        np.testing.assert_array_equal(rgb,self.rgb)
        def stop():raise InterruptedError('test cancellation')
        with self.assertRaises(InterruptedError):
            render_support_display(q,y,1.,self.chart,rgb,check_stop=stop)


if __name__=='__main__':unittest.main()
