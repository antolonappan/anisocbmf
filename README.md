# Anisotropic CMB rotation simulations and reconstruction

`sim.py` generates reproducible Gaussian CMB skies from cached lensed CAMB
spectra, applies a common anisotropic rotation, then channel miscalibration,
beam smoothing, and white noise. `test.ipynb` compares single-frequency spectra
with analytical predictions.

`test_reconstruction.ipynb` runs an eight-realization signal and zero-rotation
control test of `AlphaRecon`, with input cross-spectra, auto-spectra, estimated
N0, and a Wiener-filtered map comparison. Parameters are editable at the top;
generated reconstruction caches stay under `data/`.

`test_hilc_reconstruction.ipynb` demonstrates a toy four-frequency experiment:
HILC weights, cleaned T/E/B and noise spectra, then rotation auto- and
input cross-spectra. The mean filtering noise is estimated from a disjoint
calibration ensemble. It uses CMB plus noise, without foregrounds.

Rotation spectrum panels use raw `C_L` (no `L(L+1)/(2π)` factor) on log–log
axes. Negative measured bands are explicitly marked as negative magnitudes.
`test_hilc_random_angles.ipynb` repeats the four-frequency reconstruction with
one fixed, seeded miscalibration angle per channel, drawn uniformly from
0.1–0.2 degrees. It defaults to `nside=128` for a quick end-to-end run.

`test_masked_reconstruction.ipynb` compares a single masked channel with
`alpha=None` versus `alpha=0.2` degrees, keeping the spatial rotation and
realization seeds fixed. It uses `nside=512` and `lmax_ivf=lmax_qlm=1024`.
Place the supplied `binary_GAL_mask_N1024.fits` in the repository root:
field 2 (the third FITS column) is downgraded to 512 with fractional edges.
This local input file is not bundled with the code.

Pass `mask='binary_GAL_mask_N1024.fits', mask_field=2` to `CMBobs`, or supply
a RING HEALPix weight array. The mask multiplies T/Q/U after beam smoothing
and noise. `noise_map` includes that mask by default; `apply_mask=False`
returns the underlying full-sky noise. Mask content is included in cache
identities. The current isotropic filter is unchanged: masked reconstructions
are diagnostics without a cut-sky response, mean-field subtraction, or
mode-coupling correction. No automatic sky-fraction rescaling is applied.

`test_masked_hilc_3freq.ipynb` performs HILC on three already-masked channels
(90/150/220 GHz), with miscalibration angles 0.10/0.15/0.20 degrees. It uses
the same field-2 mask at `nside=512`, with all harmonic limits at 1024.
The exact masked noise is propagated through the HILC weights; its mean
spectra enter reconstruction filtering. Four calibration simulations and
four disjoint test simulations show weights, cleaned spectra, noise, and raw
rotation power on log–log axes. The cleaned sky is not masked a second time;
the cut-sky interpretation limits above still apply.

## Polarization rotation reconstruction

Requires NumPy, healpy, CAMB (for the first spectra calculation), and a working
[Plancklens installation](https://github.com/carronj/plancklens) including its
response/noise numerical dependencies.

```python
from sim import CMBobs
from recon import AlphaRecon

obs = CMBobs(nside=256, nlevp=[10, 15], fwhm=[30, 20],
             alpha=[0.1, 0.2], Acb=1e-5)
rec = AlphaRecon(obs, libdir='./data/recon_example', channel=0,
                 lmin_ivf=30, lmax_ivf=512, lmax_qlm=256, data_idx=1000)

alpha_lm = rec.reconstruct(0)       # normalized rotation alms, radians
noise_cl = rec.noise_spectrum(0)    # normalized semi-analytical N0, rad^2
response = rec.response()

# The underlying Plancklens libraries remain directly available:
qlm = rec.qlms_dd.get_sim_qlm('a_p', 0)  # unnormalized
response = rec.qresp_dd.get_response('a_p', 'a')
nhl = rec.nhl_dd.get_sim_nhl(0, 'a_p', 'a_p')  # unnormalized
elm_filtered = rec.ivfs.get_sim_elm(0)
```

Use a separate `AlphaRecon` for each channel, or pass the HILC combination below.
`obs.as_plancklens(channel=0, data_idx=1000)` exposes `get_sim_tmap`,
`get_sim_pmap`, and `hashdict` for use with other Plancklens filtering libraries.
Index `-1` selects `data_idx`; nonnegative indices keep their original meaning.
Reserve `data_idx` outside the Monte Carlo sample when treating it as data.

The filtering uses each channel's polarized Gaussian beam and white-noise level,
without a pixel window. The same lensed spectra (including BB) enter the QE's
Wiener leg, response, and noise weights. The fiducial covariance is unrotated;
normalization is a small-rotation, full-sky approximation. Miscalibration is
included in the reconstructed rotation, not subtracted. No mean-field or N1
correction is performed, and the semi-analytical noise is not a full RDN0
pipeline. Cache configurations are checked by Plancklens; use a new `libdir`
after changing parameters and do not mutate an observation after constructing
its reconstruction libraries.

## Harmonic ILC across frequencies

```python
from hilc import HILC

cleaned = HILC(obs, libdir='./data/hilc', lmax=512, bin_width=20)
teb = cleaned.cleaned_alms(0)        # (3, nalm), beam-deconvolved T/E/B
weights = cleaned.weights(0)        # (3, lmax+1, 1, nfreq)
noise = cleaned.noise_spectra(0)    # (3, lmax+1), residual TT/EE/BB in uK²

# Fit filtering noise on an explicit ensemble, separate from reconstruction idx=0.
noise_indices = range(100, 120)
mean_noise = cleaned.mean_noise_spectra(noise_indices)
rec = AlphaRecon(cleaned, libdir='./data/recon_hilc', noise_indices=noise_indices,
                 lmin_ivf=30, lmax_ivf=512, lmax_qlm=256)
alpha_lm = rec.reconstruct(0)
```

Following [COBI's HILC workflow](https://github.com/antolonappan/cobi/blob/0cbcb35e36f54266f9e5d5aab5456fd518bae83f/cobi/simulation/sky.py#L517),
weights are fitted to observed alms and applied unchanged to the corresponding
noise-only alms. `CMBobs.noise_map(idx)` returns exactly the noise added to its
observation. Each frequency's polarized beam is deconvolved before combining;
no pixel window is used. This implementation cleans T as well as E and B,
using separate empirical covariance matrices and `(2l+1)`-weighted bins.
Weights sum to one per field and multipole, with a regularized covariance
pseudoinverse for singular inputs.

`AlphaRecon` uses the arithmetic mean of the **noise powers**, not averaged
noise alms or total sky power:

```text
ftl = 1 / (C_TT + mean_N_TT)
fel = 1 / (C_EE + mean_N_EE)
fbl = 1 / (C_BB + mean_N_BB)
```

Multipoles below `lmin_ivf` are zeroed. HILC alms go directly into Plancklens'
harmonic filter with unit transfer functions, so beams are not removed twice.
`rec.noise_cls` exposes the actual mean spectra used. HILC products (alms,
weights, noise alms, noise spectra) are cached per simulation and configuration.
Do not mutate the input observation after creating HILC.

The unit-response constraint assumes the same calibrated CMB in each channel.
Unequal miscalibration angles are retained, and empirically fitted weights can
introduce ILC signal bias. Mean noise filtering does not by itself calibrate
these effects; reconstruction validation should use the full cleaning pipeline.

The noise library wrapper corrects versions of Plancklens whose `_get_cls`
returns the spectrum length instead of the inclusive maximum multipole.

Based on the Plancklens [idealized example](https://github.com/carronj/plancklens/blob/master/params/idealized_example.py)
and [quickstart](https://github.com/carronj/plancklens/blob/master/examples/lensingrec_quickstart.ipynb),
and the [S4bird simulation adapter](https://github.com/antolonappan/s4bird/blob/1e0cade60da958128518443c6ccaf21c63aaee26/simulation.py#L19-L48).

Run the small integration checks from the repository root:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m unittest discover -s tests -v
```
