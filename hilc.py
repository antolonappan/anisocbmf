"""Harmonic ILC and propagation of the actual simulation noise.

Follows COBI's HILC_obsEB and HILC at commit
0cbcb35e36f54266f9e5d5aab5456fd518bae83f, extending the cleaning to T/E/B.
"""
import hashlib
import json
import os
from numbers import Integral
from pathlib import Path
import tempfile

import healpy as hp
import numpy as np


class HILC:
    """Clean CMBobs frequencies with separate binned T, E, B ILC weights.

    Each frequency is beam-deconvolved before estimating the covariance.
    Weights minimize empirical variance subject to sum(w)=1 in each bin.
    The same observed-sky weights are applied to the exact noise realization.
    Output alms have unit beam and no pixel window, in uK.

    The common-CMB constraint assumes calibrated channels. Existing unequal
    miscalibration angles are kept, not corrected. Empirical ILC weights can
    suppress signal through chance correlations; an isotropic QE response
    does not correct this ILC bias.
    """

    def __init__(self, obs, libdir='./data/hilc', lmax=None, bin_width=20,
                 rcond=1e-12):
        self.obs = obs
        self.nside = obs.nside
        self.lmax = min(2 * obs.nside, obs.lmax) if lmax is None else lmax
        if (isinstance(self.lmax, bool) or not isinstance(self.lmax, Integral)
                or not 2 <= self.lmax <= obs.lmax):
            raise ValueError('lmax must be an integer between 2 and obs.lmax')
        self.lmax = int(self.lmax)
        if (isinstance(bin_width, bool) or not isinstance(bin_width, Integral)
                or bin_width < 1):
            raise ValueError('bin_width must be a positive integer')
        if not np.isscalar(rcond) or not np.isfinite(rcond) or not 0 < rcond < 1:
            raise ValueError('rcond must lie between zero and one')
        self.bin_width = int(bin_width)
        self.rcond = float(rcond)
        self.spectra = obs.spectra[:self.lmax + 1].copy()
        # Keep T monopole/dipole separate from polarization's supported modes.
        self.bins = np.r_[0, 2, np.arange(2 + self.bin_width, self.lmax + 1,
                                        self.bin_width), self.lmax + 1]
        self.beams = np.array([
            hp.gauss_beam(np.deg2rad(fwhm / 60), lmax=self.lmax, pol=True)[:, :3].T
            for fwhm in obs.fwhm
        ])
        if np.any(self.beams <= 0):
            raise ValueError('Beam underflows; reduce lmax or beam widths')
        # Configuration-specific caches, including input spectra and seed scheme.
        self._config = {
            'library': 'HILC-v1', 'obs': obs.as_plancklens().hashdict(),
            'lmax': self.lmax, 'bin_width': self.bin_width, 'rcond': self.rcond,
        }
        tag = hashlib.sha256(json.dumps(self._config, sort_keys=True).encode()).hexdigest()
        self.libdir = Path(libdir) / tag[:20]
        self.libdir.mkdir(parents=True, exist_ok=True)

    def hashdict(self):
        return self._config.copy()

    def _validate_alms(self, alms):
        alms = np.asarray(alms, dtype=complex)
        if (alms.ndim != 3 or alms.shape[0] < 1 or alms.shape[1] < 1
                or hp.Alm.getlmax(alms.shape[-1]) < 0
                or not np.isfinite(alms).all()):
            raise ValueError('alms must have shape (nfreq, nfield, nalm), finite and packed')
        return alms

    def harmonic_ilc_alm(self, alms, lbins=None):
        """Return (cleaned, weights), with COBI-compatible leading component axis.

        alms: (nfreq, nfield, nalm); cleaned: (1, nfield, nalm);
        weights: (nfield, lmax+1, 1, nfreq). Bins are half-open and must
        cover [0, lmax+1). If omitted, use this instance's bins.
        """
        alms = self._validate_alms(alms)
        nfreq, nfield, _ = alms.shape
        lmax = hp.Alm.getlmax(alms.shape[-1])
        edges = np.asarray(self.bins if lbins is None else lbins)
        if (edges.ndim != 1 or len(edges) < 2 or edges.dtype.kind not in 'iu'
                or edges[0] != 0 or edges[-1] != lmax + 1
                or np.any(np.diff(edges) <= 0)):
            raise ValueError('bins must be increasing integer edges covering [0, lmax+1)')
        weights = np.empty((nfield, lmax + 1, 1, nfreq))
        for field in range(nfield):
            cov = np.empty((lmax + 1, nfreq, nfreq))
            for i in range(nfreq):
                for j in range(i + 1):
                    cov[:, i, j] = cov[:, j, i] = hp.alm2cl(alms[i, field], alms[j, field])
            for lo, hi in zip(edges[:-1], edges[1:]):
                covariance = np.average(cov[lo:hi], axis=0, weights=2*np.arange(lo, hi)+1)
                variance = np.diag(covariance)
                zero = variance <= 0
                if np.any(zero):
                    # Exact zero variance: any mixture of these channels is optimal.
                    w = zero.astype(float) / zero.sum()
                else:
                    inv_std = 1 / np.sqrt(variance)
                    correlation = covariance * inv_std[:, None] * inv_std[None, :]
                    inv_cov = (np.linalg.pinv(correlation, rcond=self.rcond, hermitian=True)
                               * inv_std[:, None] * inv_std[None, :])
                    numerator = inv_cov.sum(axis=1)
                    denominator = numerator.sum()
                    w = (numerator / denominator if denominator > 0
                         else np.full(nfreq, 1 / nfreq))
                weights[field, lo:hi, 0] = w
        return self.apply_harmonic_W(weights, alms), weights

    def apply_harmonic_W(self, weights, alms):
        """Apply the SAME weights to sky or noise alms (no weight refitting)."""
        alms = self._validate_alms(alms)
        nfreq, nfield, nalm = alms.shape
        ell, _ = hp.Alm.getlm(hp.Alm.getlmax(nalm))
        weights = np.asarray(weights)
        if weights.shape != (nfield, ell.max() + 1, 1, nfreq):
            raise ValueError('weights and alms have incompatible shapes')
        cleaned = np.zeros((1, nfield, nalm), dtype=complex)
        for field in range(nfield):
            for channel in range(nfreq):
                cleaned[0, field] += alms[channel, field] * weights[field, ell, 0, channel]
        return cleaned

    def _deconvolved_alms(self, maps):
        alms = np.array([hp.map2alm(sky, lmax=self.lmax, pol=True, iter=3)
                         for sky in maps])
        for channel in range(len(alms)):
            for field in range(3):
                hp.almxfl(alms[channel, field], 1 / self.beams[channel, field], inplace=True)
        return alms

    def _products(self, idx):
        idx = self.obs._validate_idx(idx)
        path = self.libdir / f'sim_{idx:04d}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}
        alms = self._deconvolved_alms(self.obs.observedmap(idx))
        noise_alms = self._deconvolved_alms(self.obs.noise_map(idx))
        cleaned, weights = self.harmonic_ilc_alm(alms)
        noise = self.apply_harmonic_W(weights, noise_alms)[0]
        products = dict(alms=cleaned[0], weights=weights, noise_alms=noise,
                        noise_cls=np.array([hp.alm2cl(alm) for alm in noise]))
        # Atomic replace also makes simultaneous deterministic requests safe.
        with tempfile.NamedTemporaryFile(dir=self.libdir, suffix='.npz', delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            np.savez(temporary, **products)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return products

    def cleaned_alms(self, idx):
        """Beam-deconvolved T/E/B alms, shape (3, nalm)."""
        return self._products(idx)['alms']

    def weights(self, idx):
        return self._products(idx)['weights']

    def noise_alms(self, idx):
        return self._products(idx)['noise_alms']

    def noise_spectra(self, idx):
        """Residual TT/EE/BB noise C_l in uK^2, shape (3, lmax+1)."""
        return self._products(idx)['noise_cls']

    def mean_noise_spectra(self, indices):
        """Arithmetic mean of cleaned noise POWERS over specified realizations."""
        indices = tuple(self.obs._validate_idx(idx) for idx in indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError('noise indices must be nonempty and unique')
        total = np.zeros((3, self.lmax + 1))
        for idx in indices:
            total += self.noise_spectra(idx)
        return total / len(indices)

    def as_plancklens(self, channel=0, data_idx=0, lmax=None, harmonic=False):
        if isinstance(channel, bool) or channel != 0:
            raise ValueError('HILC has a single cleaned component; channel must be 0')
        return HILCLibrary(self, data_idx, lmax, harmonic)


class HILCLibrary:
    """Plancklens input adapter for cleaned maps or already-deconvolved alms.

    The harmonic filter still calls get_sim_tmap/get_sim_pmap, but expects
    T/E/B alms from those methods when harmonic=True.
    """

    def __init__(self, hilc, data_idx=0, lmax=None, harmonic=False):
        self.hilc = hilc
        self.channel = 0
        self.data_idx = hilc.obs._validate_idx(data_idx)
        self.lmax = hilc.lmax if lmax is None else lmax
        if (isinstance(self.lmax, bool) or not isinstance(self.lmax, Integral)
                or not 2 <= self.lmax <= hilc.lmax):
            raise ValueError('adapter lmax must lie between 2 and HILC lmax')
        self.harmonic = harmonic
        self.filter_hash = ''
        ell, m = hp.Alm.getlm(self.lmax)
        self._alm_indices = hp.Alm.getidx(hilc.lmax, ell, m)

    def hashdict(self):
        return dict(hilc=self.hilc.hashdict(), data_idx=self.data_idx,
                    lmax=self.lmax, harmonic=self.harmonic, filter_hash=self.filter_hash)

    def _alms(self, idx):
        if isinstance(idx, bool) or not isinstance(idx, Integral):
            raise ValueError('idx must be -1 or a nonnegative integer')
        idx = self.data_idx if idx == -1 else idx
        return self.hilc.cleaned_alms(idx)[:, self._alm_indices]

    def get_sim_tmap(self, idx):
        alm = self._alms(idx)[0]
        return alm if self.harmonic else hp.alm2map(alm, self.hilc.nside)

    def get_sim_pmap(self, idx):
        alms = self._alms(idx)[1:]
        return alms if self.harmonic else hp.alm2map_spin(alms, self.hilc.nside, 2, self.lmax)
