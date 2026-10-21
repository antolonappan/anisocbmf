# Anisotropic CMB rotation simulations and reconstruction

`sim.py` generates reproducible Gaussian CMB skies from cached lensed CAMB
spectra, applies a common anisotropic rotation, then channel miscalibration,
beam smoothing, and white noise. `test.ipynb` compares single-frequency spectra
with analytical predictions.

`test_reconstruction.ipynb` runs an eight-realization signal and zero-rotation
control test of `AlphaRecon`, with input cross-spectra, auto-spectra, estimated
N0, and a Wiener-filtered map comparison. Parameters are editable at the top;
generated reconstruction caches stay under `data/`.

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

Use a separate `AlphaRecon` for each channel; this does not combine frequencies.
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

The noise library wrapper corrects versions of Plancklens whose `_get_cls`
returns the spectrum length instead of the inclusive maximum multipole.

Based on the Plancklens [idealized example](https://github.com/carronj/plancklens/blob/master/params/idealized_example.py)
and [quickstart](https://github.com/carronj/plancklens/blob/master/examples/lensingrec_quickstart.ipynb),
and the [S4bird simulation adapter](https://github.com/antolonappan/s4bird/blob/1e0cade60da958128518443c6ccaf21c63aaee26/simulation.py#L19-L48).

Run the small integration checks from the repository root:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m unittest discover -s tests -v
```
