"""HILC constraints, beam/noise propagation, and real Plancklens integration."""
import tempfile
import unittest
from unittest.mock import patch

import healpy as hp
import numpy as np

from sim import CMBobs
from hilc import HILC
from recon import AlphaRecon


class TestHILC(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_known_minimum_variance_and_singular_covariance(self):
        hilc = HILC(CMBobs(8, [2, 4], [0, 0]), self.tmp.name, lmax=4)
        # Orthogonal modes give exactly diagonal frequency covariance 1:4.
        alms = np.zeros((2, 1, hp.Alm.getsize(4)), dtype=complex)
        alms[0, 0, hp.Alm.getidx(4, 2, 0)] = 1
        alms[1, 0, hp.Alm.getidx(4, 3, 0)] = 2
        clean, weights = hilc.harmonic_ilc_alm(alms, [0, 5])
        np.testing.assert_allclose(weights[0, :, 0], np.tile([0.8, 0.2], (5, 1)))
        np.testing.assert_allclose(clean[0], 0.8*alms[0] + 0.2*alms[1])
        common = np.repeat(alms[:1], 2, axis=0)
        clean, weights = hilc.harmonic_ilc_alm(common, [0, 5])
        np.testing.assert_allclose(weights.sum(axis=-1), 1)
        np.testing.assert_allclose(clean[0], common[0])
        clean, weights = hilc.harmonic_ilc_alm(common * 0, [0, 5])
        self.assertTrue(np.isfinite(weights).all())
        self.assertFalse(np.any(clean))
        with self.assertRaises(ValueError):
            hilc.harmonic_ilc_alm(alms, [2, 5])

    def test_noise_is_exact_and_weights_propagate_linearly(self):
        obs = CMBobs(16, [2, 4], [20, 30], [0, 0.1], Acb=1e-5)
        for channel in range(2):
            rng = np.random.default_rng(np.random.SeedSequence([7, 1, channel]))
            expected = rng.normal(size=(3, obs.npix)) * obs._noise_sigma[channel, :, None]
            np.testing.assert_array_equal(obs.noise_map(7, channel), expected)
        with patch.object(obs, 'cmbrotated', return_value=np.zeros((3, obs.npix))):
            np.testing.assert_array_equal(obs.observedmap(7), obs.noise_map(7))
        hilc = HILC(obs, self.tmp.name, lmax=24, bin_width=8)
        observed = hilc._deconvolved_alms(obs.observedmap(0))
        noise = hilc._deconvolved_alms(obs.noise_map(0))
        weights = hilc.weights(0)
        np.testing.assert_allclose(weights.sum(axis=-1), 1, atol=1e-12)
        np.testing.assert_allclose(hilc.noise_alms(0), hilc.apply_harmonic_W(weights, noise)[0])
        np.testing.assert_allclose(
            hilc.cleaned_alms(0) - hilc.noise_alms(0),
            hilc.apply_harmonic_W(weights, observed - noise)[0], atol=1e-12,
        )
        with patch.object(obs, 'observedmap', side_effect=AssertionError('cache missed')):
            self.assertTrue(np.isfinite(hilc.cleaned_alms(0)).all())

    def test_single_channel_beam_and_noise(self):
        obs = CMBobs(16, 3, 30)
        hilc = HILC(obs, self.tmp.name, lmax=24)
        beam = hp.gauss_beam(np.deg2rad(0.5), lmax=24, pol=True)[:, :3].T
        sky = np.array(hp.map2alm(obs.observedmap(0)[0], lmax=24, iter=3))
        expected = np.array([hp.almxfl(sky[i], 1/beam[i]) for i in range(3)])
        np.testing.assert_allclose(hilc.cleaned_alms(0), expected)
        np.testing.assert_array_equal(hilc.weights(0), 1)
        measured = hilc.mean_noise_spectra(range(30))
        nlev = np.array([obs.nlevt[0], obs.nlevp[0], obs.nlevp[0]])
        theory = (nlev[:, None] * np.pi / (180*60) / beam)**2
        # Broad multipole average of independent noise realizations.
        np.testing.assert_allclose((measured[:, 2:] / theory[:, 2:]).mean(axis=1), 1, rtol=0.08)
        with self.assertRaises(ValueError):
            hilc.mean_noise_spectra([])
        with self.assertRaises(ValueError):
            hilc.mean_noise_spectra([0, 0])

    def test_mean_noise_in_all_filters_and_reconstruction(self):
        obs = CMBobs(16, [2, 4], [20, 30], Acb=1e-5)
        hilc = HILC(obs, self.tmp.name + '/hilc', lmax=24, bin_width=8)
        indices = [10, 11, 12]
        mean = np.mean([hilc.noise_spectra(idx) for idx in indices], axis=0)
        np.testing.assert_array_equal(hilc.mean_noise_spectra(indices), mean)
        rec = AlphaRecon(hilc, self.tmp.name + '/rec', noise_indices=indices,
                         lmin_ivf=4, lmax_ivf=20, lmax_qlm=8)
        for i, (key, getter) in enumerate(zip(('tt', 'ee', 'bb'),
                (rec.ivfs.get_ftl, rec.ivfs.get_fel, rec.ivfs.get_fbl))):
            np.testing.assert_array_equal(rec.noise_cls[key], mean[i, :21])
            np.testing.assert_allclose(getter()[4:], 1/(rec.cl_len[key][4:] + mean[i, 4:21]))
            np.testing.assert_array_equal(getter()[:4], 0)
        for beam in rec.transf.values():
            np.testing.assert_array_equal(beam, 1)
        self.assertEqual(rec.ivfs.get_sim_elm(0).shape, (hp.Alm.getsize(20),))
        self.assertTrue(np.isfinite(rec.reconstruct(0)).all())
        self.assertTrue(np.isfinite(rec.noise_spectrum(0)).all())
        with self.assertRaises(ValueError):
            AlphaRecon(hilc, self.tmp.name + '/bad')
        changed = HILC(CMBobs(16, [3, 4], [20, 30]), self.tmp.name + '/hilc', lmax=24)
        self.assertNotEqual(hilc.libdir, changed.libdir)


if __name__ == '__main__':
    unittest.main()
