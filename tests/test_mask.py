import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import healpy as hp
import numpy as np
from sim import CMBobs


class TestMask(unittest.TestCase):
    def test_final_mask_and_exact_noise(self):
        mask = np.linspace(0, 1, hp.nside2npix(8))
        kwargs = dict(nside=8, nlevp=[2, 3], fwhm=[10, 30], alpha=[0.1, 0.2], Acb=1e-5)
        full = CMBobs(**kwargs)
        cut = CMBobs(**kwargs, mask=mask)
        np.testing.assert_array_equal(cut.observedmap(3), full.observedmap(3) * mask)
        np.testing.assert_array_equal(cut.noise_map(3), full.noise_map(3) * mask)
        np.testing.assert_array_equal(cut.noise_map(3, apply_mask=False), full.noise_map(3))
        # Fractional weights catch accidentally masking noise twice.
        with patch.object(cut, 'cmbrotated', return_value=np.zeros((3, cut.npix))):
            np.testing.assert_array_equal(cut.observedmap(3), cut.noise_map(3))
        self.assertNotEqual(cut.as_plancklens().hashdict(), full.as_plancklens().hashdict())

    def test_fits_field_and_downgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'mask.fits'
            field2 = np.arange(hp.nside2npix(16)) % 2
            hp.write_map(str(path), [np.zeros_like(field2), np.ones_like(field2), field2],
                         dtype=float, overwrite=True)
            cut = CMBobs(8, 1, 0, mask=path)
            np.testing.assert_array_equal(cut.mask, hp.ud_grade(field2.astype(float), 8))
            self.assertTrue(np.any((cut.mask > 0) & (cut.mask < 1)))
        for invalid in [np.zeros(12), np.full(12, -1.), np.full(12, np.nan), np.ones(13)]:
            with self.assertRaises(ValueError):
                CMBobs(8, 1, 0, mask=invalid)


if __name__ == '__main__':
    unittest.main()
