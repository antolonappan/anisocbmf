"""Small real Plancklens integration checks; run with unittest discovery."""
import tempfile
import unittest
from pathlib import Path

import healpy as hp
import numpy as np

from sim import CMBobs
from recon import AlphaRecon


class TestAlphaRecon(unittest.TestCase):
    def test_adapter_and_channels(self):
        obs = CMBobs(8, [1, 2], [10, 20], [0, 0.1], Acb=1e-5)
        lib = obs.as_plancklens(channel=1, data_idx=3)
        expected = obs.observedmap(3)[1]
        np.testing.assert_array_equal(lib.get_sim_tmap(-1), expected[0])
        np.testing.assert_array_equal(lib.get_sim_pmap(-1), expected[1:])
        self.assertNotEqual(lib.hashdict(), obs.as_plancklens(0, 3).hashdict())
        with self.assertRaises(ValueError):
            obs.as_plancklens(2)
        with self.assertRaises(ValueError):
            lib.get_sim_pmap(-2)

    def test_reconstruction_response_noise_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            obs = CMBobs(32, 1, 10, alpha=0.1)
            rec = AlphaRecon(obs, directory, lmax_ivf=48, lmax_qlm=16)
            alm = rec.reconstruct(0)
            self.assertEqual(alm.shape, (hp.Alm.getsize(16),))
            self.assertTrue(np.isfinite(alm).all())
            # A known positive uniform rotation fixes sign and radian units.
            mean_degrees = np.rad2deg(alm[0].real / np.sqrt(4 * np.pi))
            self.assertAlmostEqual(mean_degrees, 0.1, delta=0.03)
            noise = rec.noise_spectrum(0)
            self.assertEqual(noise.shape, (17,))
            self.assertTrue(np.all(np.isfinite(noise) & (noise >= 0)))
            np.testing.assert_allclose(
                alm, hp.almxfl(rec.qlms_dd.get_sim_qlm('a_p', 0),
                              rec.normalization()), rtol=1e-6,
            )
            again = AlphaRecon(obs, directory, lmax_ivf=48, lmax_qlm=16)
            np.testing.assert_allclose(again.reconstruct(0), alm, rtol=1e-6)
            np.testing.assert_array_equal(again.noise_spectrum(0), noise)
            # Full-sky transfer functions match polarized smoothing, no pixwin.
            beam = hp.gauss_beam(np.deg2rad(10/60), lmax=48, pol=True)
            np.testing.assert_array_equal(rec.transf['e'], beam[:, 1])
            np.testing.assert_array_equal(rec.ivfs.get_fel()[:2], 0)
            self.assertTrue(list(Path(directory).rglob('*.fits')))

    def test_anisotropic_input(self):
        with tempfile.TemporaryDirectory() as directory:
            obs = CMBobs(16, [2, 3], [20, 30], [0, 0.1], Acb=1e-5)
            rec = AlphaRecon(obs, directory, channel=1, lmax_ivf=24, lmax_qlm=8)
            self.assertTrue(np.isfinite(rec.reconstruct(1)).all())
            self.assertTrue(np.isfinite(rec.noise_spectrum(1)).all())
            self.assertEqual(rec.sims.channel, 1)
            with self.assertRaises(ValueError):
                AlphaRecon(obs, directory, lmax_ivf=obs.lmax + 1)


if __name__ == '__main__':
    unittest.main()
