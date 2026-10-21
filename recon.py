"""Full-sky polarization rotation reconstruction using Plancklens.

Adapted from params/idealized_example.py and examples/lensingrec_quickstart.ipynb
at https://github.com/carronj/plancklens .
"""

from numbers import Integral
from pathlib import Path

import healpy as hp
import numpy as np


class AlphaRecon:
    """Isotropic filtering and the polarization-only rotation QE ('a_p').

    obs is a CMBobs instance. Select one frequency with channel; frequencies
    are not coadded. lmin_ivf/lmax_ivf select CMB modes and lmax_qlm selects
    reconstructed rotation modes. Cache directories must not be reused for
    different configurations (Plancklens checks their hashes).

    The fiducial model is the unrotated lensed CMB with homogeneous noise.
    The response is linear in small rotation, including any miscalibration;
    miscalibration is not removed. No mean field or N1 subtraction is done.
    """

    key = 'a_p'

    def __init__(self, obs, libdir='./data/alpha_recon', channel=0,
                 lmin_ivf=2, lmax_ivf=None, lmax_qlm=None, data_idx=0):
        from plancklens import nhl, qest, qresp, utils
        from plancklens.filt import filt_simple

        self.obs = obs
        self.sims = obs.as_plancklens(channel=channel, data_idx=data_idx)
        self.channel = self.sims.channel
        self.nside = obs.nside
        # Stay below the highest HEALPix modes by default.
        self.lmax_ivf = self._multipole(
            min(2 * obs.nside, obs.lmax) if lmax_ivf is None else lmax_ivf,
            'lmax_ivf', 2, obs.lmax,
        )
        self.lmin_ivf = self._multipole(lmin_ivf, 'lmin_ivf', 2, self.lmax_ivf)
        self.lmax_qlm = self._multipole(
            self.lmax_ivf if lmax_qlm is None else lmax_qlm,
            'lmax_qlm', 0, min(2 * self.lmax_ivf, 3 * obs.nside - 1),
        )
        self.libdir = Path(libdir) / f'freq_{self.channel}'
        self.cl_len = {key: obs.spectra[:self.lmax_ivf + 1, i].copy()
                       for i, key in enumerate(('tt', 'ee', 'bb', 'te'))}
        # Keep lensed BB in ALL QE weights, including ivfs' Wiener leg.
        # Zeroing BB only in nhl/qresp would normalize a different estimator.
        self.cl_weight = {key: cl.copy() for key, cl in self.cl_len.items()}
        beam = hp.gauss_beam(np.deg2rad(obs.fwhm[self.channel] / 60),
                             lmax=self.lmax_ivf, pol=True)
        self.transf = {key: beam[:, i] for i, key in enumerate(('t', 'e', 'b'))}
        # No pixel window: CMBobs does not apply one.
        nlev = {'t': obs.nlevt[self.channel],
                'e': obs.nlevp[self.channel], 'b': obs.nlevp[self.channel]}
        filters = {}
        for key in ('t', 'e', 'b'):
            if np.any(self.transf[key] <= 0):
                raise ValueError('Beam underflows in the selected multipole range')
            noise = (nlev[key] * np.pi / (180 * 60) / self.transf[key])**2
            filters[key] = utils.cli(self.cl_len[key * 2] + noise)
            filters[key][:self.lmin_ivf] = 0

        self.ivfs = filt_simple.library_fullsky_sepTP(
            str(self.libdir / 'ivfs'), self.sims, self.nside, self.transf,
            self.cl_weight, filters['t'], filters['e'], filters['b'], cache=True,
        )
        self.qlms_dd = qest.library_sepTP(
            str(self.libdir / 'qlms_dd'), self.ivfs, self.ivfs,
            self.cl_weight['te'], self.nside, lmax_qlm=self.lmax_qlm,
        )
        class NhlLibrary(nhl.nhl_lib_simple):
            def _get_cls(self, idx, spins):
                cls, _ = super()._get_cls(idx, spins)
                # Some Plancklens versions return len(C_l), not lmax here.
                # Derive the inclusive maximum to avoid an extra QE mode.
                return cls, len(next(iter(cls.values()))) - 1

        self.nhl_dd = NhlLibrary(
            str(self.libdir / 'nhl_dd'), self.ivfs, self.cl_weight, self.lmax_qlm,
        )
        self.qresp_dd = qresp.resp_lib_simple(
            str(self.libdir / 'qresp_dd'), self.lmax_ivf, self.cl_weight,
            self.cl_len, {'t': self.ivfs.get_ftl(), 'e': self.ivfs.get_fel(),
                          'b': self.ivfs.get_fbl()}, self.lmax_qlm,
        )

    @staticmethod
    def _multipole(value, name, lower, upper):
        if (isinstance(value, bool) or not isinstance(value, Integral)
                or not lower <= value <= upper):
            raise ValueError(f'{name} must be an integer in [{lower}, {upper}]')
        return int(value)

    def response(self):
        """Return the unnormalized a_p response to rotation source 'a'."""
        return self.qresp_dd.get_response(self.key, 'a')

    def normalization(self):
        """Inverse response; modes with zero response are set to zero."""
        from plancklens import utils
        return utils.cli(self.response())

    def reconstruct(self, idx):
        """Return normalized rotation alms in radians (no mean-field removal)."""
        qlm = self.qlms_dd.get_sim_qlm(self.key, idx)
        return hp.almxfl(qlm, self.normalization())

    def noise_spectrum(self, idx):
        """Normalized semi-analytical disconnected noise estimate in rad^2."""
        return (self.nhl_dd.get_sim_nhl(idx, self.key, self.key)
                * self.normalization()**2)
