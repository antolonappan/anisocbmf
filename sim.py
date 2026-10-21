import pickle
import hashlib
from numbers import Integral
from pathlib import Path

import healpy as hp
import numpy as np


def get_lensed_cmb_spectra(lmax=6000):
    """Return lensed CMB C_ell in microkelvin squared, for ell=0..lmax.

    Columns are TT, EE, BB, TE; the returned shape is (lmax + 1, 4).
    On the first call, compute and cache all multipoles through 6000 in
    ./data/spectra.pkl, relative to the current working directory. Later
    calls load that array and slice it without running CAMB again.

    The cache uses the fixed fiducial cosmology below. Delete spectra.pkl
    to recompute it after changing the cosmology.
    """
    cache_lmax = 6000
    if isinstance(lmax, bool) or not isinstance(lmax, Integral):
        raise TypeError("lmax must be an integer")
    if not 0 <= lmax <= cache_lmax:
        raise ValueError(f"lmax must be between 0 and {cache_lmax}")

    cache_path = Path("data") / "spectra.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            spectra = pickle.load(handle)
    else:
        import camb

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        params = camb.CAMBparams()
        params.set_cosmology(
            H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06
        )
        params.InitPower.set_params(As=2e-9, ns=0.965, r=0)
        params.set_for_lmax(cache_lmax, lens_potential_accuracy=1)
        results = camb.get_results(params)
        spectra = results.get_lensed_scalar_cls(
            lmax=cache_lmax, CMB_unit="muK", raw_cl=True
        )
        with cache_path.open("wb") as handle:
            pickle.dump(spectra, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return spectra[: lmax + 1]


class CMBobs:
    """Simulate a common CMB sky observed in one or more frequency channels.

    nlevp is the Q/U white-noise level in uK-arcmin; nlevt = nlevp/sqrt(2).
    fwhm is the Gaussian beam width in arcminutes. Both accept scalars or
    nonempty 1-D sequences of equal length (a scalar means one channel).
    alpha is None, or matching per-channel angles in degrees, with convention
    Q' = Q*cos(2*alpha) - U*sin(2*alpha),
    U' = Q*sin(2*alpha) + U*cos(2*alpha).

    Acb is the nonnegative anisotropic birefringence amplitude in radians
    squared: L*(L+1)*C_L^aa/(2*pi) = Acb for L >= 2. None or zero disables
    it. A common spatial rotation is applied before channel miscalibration,
    beam convolution, and noise. alpha_map returns radians, whereas the
    per-channel alpha argument above is in degrees.

    Maps contain T, Q, U in uK in HEALPix RING order. The simulation uses
    lmax = min(3*nside - 1, 6000), with no pixel window applied.
    """

    def __init__(self, nside, nlevp, fwhm, alpha=None, Acb=0.0):
        if (isinstance(nside, bool) or not isinstance(nside, Integral)
                or not hp.isnsideok(nside)):
            raise ValueError("nside must be a valid positive HEALPix integer")
        self.nside = int(nside)
        self.npix = hp.nside2npix(self.nside)
        self.lmax = min(3 * self.nside - 1, 6000)
        amplitude = np.asarray(0.0 if Acb is None else Acb, dtype=float)
        if amplitude.ndim != 0 or not np.isfinite(amplitude) or amplitude < 0:
            raise ValueError("Acb must be a finite nonnegative scalar or None")
        self.Acb = float(amplitude)
        self.nlevp = self._channel_array(nlevp, "nlevp")
        self.fwhm = self._channel_array(fwhm, "fwhm")
        if self.nlevp.size != self.fwhm.size:
            raise ValueError("nlevp and fwhm must have the same length")
        if np.any(self.nlevp < 0) or np.any(self.fwhm < 0):
            raise ValueError("nlevp and fwhm must be nonnegative")
        self.nfreq = self.nlevp.size
        self.alpha = (np.zeros(self.nfreq) if alpha is None
                      else self._channel_array(alpha, "alpha"))
        if self.alpha.size != self.nfreq:
            raise ValueError("alpha must have the same length as nlevp and fwhm")
        self.nlevt = self.nlevp / np.sqrt(2)
        pixel_size_arcmin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        self._noise_sigma = np.column_stack(
            (self.nlevt, self.nlevp, self.nlevp)
        ) / pixel_size_arcmin
        self.spectra = get_lensed_cmb_spectra(self.lmax)

    @staticmethod
    def _channel_array(value, name):
        array = np.atleast_1d(np.array(value, dtype=float, copy=True))
        if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite scalar or nonempty 1-D array")
        return array

    @staticmethod
    def _validate_idx(idx):
        if isinstance(idx, bool) or not isinstance(idx, Integral) or idx < 0:
            raise ValueError("idx must be a nonnegative integer")
        return int(idx)

    def cmblensed(self, idx):
        """Return an unrotated, unsmoothed T/Q/U sky of shape (3, npix).

        synfast draws a Gaussian sky from the lensed spectra. This is not an
        explicit lensing remapping and does not reproduce lensing higher-order
        correlations. The caller's NumPy random state is restored afterward.
        """
        idx = self._validate_idx(idx)
        seed = np.random.SeedSequence([idx, 0]).generate_state(1)[0]
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            return hp.synfast(
                self.spectra.T, self.nside, lmax=self.lmax,
                pol=True, new=True, pixwin=False,
            )
        finally:
            np.random.set_state(state)

    def cl_aa(self):
        """Return C_L^aa in rad^2, shape (lmax + 1,), with L=0,1 zero.

        Uses the scale-invariant convention in COBI's CMB implementation:
        https://github.com/antolonappan/cobi/blob/main/cobi/simulation/cmb.py
        """
        cls = np.zeros(self.lmax + 1)
        ell = np.arange(2, self.lmax + 1, dtype=float)
        cls[2:] = 2 * np.pi * self.Acb / (ell * (ell + 1))
        return cls

    def alpha_alm(self, idx):
        """Return reproducible scalar rotation alms, independent of CMB/noise."""
        idx = self._validate_idx(idx)
        if self.Acb == 0:
            return np.zeros(hp.Alm.getsize(self.lmax), dtype=complex)
        seed = np.random.SeedSequence([idx, 2]).generate_state(1)[0]
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            return hp.synalm(self.cl_aa(), lmax=self.lmax, new=True)
        finally:
            np.random.set_state(state)

    def alpha_map(self, idx):
        """Return the common spatial rotation in radians, RING shape (npix,)."""
        idx = self._validate_idx(idx)
        if self.Acb == 0:
            return np.zeros(self.npix)
        return hp.alm2map(self.alpha_alm(idx), self.nside, lmax=self.lmax)

    def cmbrotated(self, idx):
        """Return lensed T/Q/U after spatial rotation, before instrument effects.

        This rotates the Gaussian sky drawn from lensed CMB spectra; it does
        not implement COBI's rotation-before-lensing remapping with lenspyx.
        """
        cmb = self.cmblensed(idx)
        if self.Acb == 0:
            return cmb
        angle = 2 * self.alpha_map(idx)
        c, s = np.cos(angle), np.sin(angle)
        sky = cmb.copy()
        sky[1] = c * cmb[1] - s * cmb[2]
        sky[2] = s * cmb[1] + c * cmb[2]
        return sky

    def observedmap(self, idx):
        """Return shape (nfreq, 3, npix), including for a single channel.

        Apply the shared spatial rotation, then channel miscalibration,
        convolve with each channel's beam, and add independent white noise.
        CMB, spatial rotation, and channel noise use separate reproducible
        random streams keyed by idx (and channel index for noise).
        """
        idx = self._validate_idx(idx)
        cmb = self.cmbrotated(idx)
        maps = np.empty((self.nfreq, 3, self.npix))
        for channel in range(self.nfreq):
            angle = 2 * np.deg2rad(self.alpha[channel])
            c, s = np.cos(angle), np.sin(angle)
            sky = cmb.copy()
            sky[1] = c * cmb[1] - s * cmb[2]
            sky[2] = s * cmb[1] + c * cmb[2]
            if self.fwhm[channel] > 0:
                sky = hp.smoothing(
                    sky, fwhm=np.deg2rad(self.fwhm[channel] / 60),
                    pol=True, lmax=self.lmax,
                )
            rng = np.random.default_rng(np.random.SeedSequence([idx, 1, channel]))
            maps[channel] = sky + rng.normal(size=(3, self.npix)) * (
                self._noise_sigma[channel, :, None]
            )
        return maps

    def as_plancklens(self, channel=0, data_idx=0):
        """Expose one frequency to Plancklens; its index -1 maps to data_idx."""
        return CMBMapLibrary(self, channel, data_idx)


class CMBMapLibrary:
    """Single-frequency Plancklens map interface, following S4bird's adapter.

    https://github.com/antolonappan/s4bird/blob/1e0cade60da958128518443c6ccaf21c63aaee26/simulation.py
    data_idx identifies a simulation to treat as 'data' (-1). Use a separate
    realization outside your Monte Carlo set when estimating biases.
    """

    def __init__(self, obs, channel=0, data_idx=0):
        if (isinstance(channel, bool) or not isinstance(channel, Integral)
                or not 0 <= channel < obs.nfreq):
            raise ValueError("channel must index an existing frequency")
        self.obs = obs
        self.channel = int(channel)
        self.data_idx = obs._validate_idx(data_idx)
        self.nside = obs.nside

    def hashdict(self):
        obs = self.obs
        return {
            'sim_lib': 'CMBobs-v1', 'nside': obs.nside, 'lmax': obs.lmax,
            'channel': self.channel, 'data_idx': self.data_idx, 'Acb': obs.Acb,
            'nlevp': tuple(obs.nlevp), 'fwhm': tuple(obs.fwhm),
            'alpha': tuple(obs.alpha),
            'spectra': hashlib.sha256(obs.spectra.tobytes()).hexdigest(),
        }

    def _map(self, idx):
        if isinstance(idx, bool) or not isinstance(idx, Integral):
            raise ValueError("idx must be -1 or a nonnegative integer")
        return self.obs.observedmap(self.data_idx if idx == -1 else idx)[self.channel]

    def get_sim_tmap(self, idx):
        return self._map(idx)[0]

    def get_sim_pmap(self, idx):
        maps = self._map(idx)
        return maps[1], maps[2]
