"""MPI driver: masked-sky rotation reconstruction for a grid of (Acb, alpha).

Each MPI rank processes the simulation indices idx = first_idx + rank,
first_idx + rank + size, ... and, for each of them, every (Acb, alpha)
configuration. All configurations share the same CMB, rotation-field and
noise seeds, so they differ only in the constant miscalibration alpha and the
anisotropic amplitude Acb. Run from anywhere, e.g.

    mpirun -np 100 python src/run_sims.py --libdir /path/to/libdir

Outputs, below <libdir>/<tag>/ (tag encodes the experiment and mask):
    filters/                      shared cinv filters (dense preconditioners)
    recon/<case>/                 Plancklens caches (filtered maps, qlms, N0)
    results/<case>/sim_XXXX.npz   per-realization spectra
    summary_Acb_<Acb>.npz         all realizations and alphas stacked
<case> is Acb_<Acb>/alpha_<alpha>. Finished realizations are skipped on
restart. Spectra are f_sky corrected as in Plancklens' qecl: the
reconstruction carries two powers of the mask, so auto spectra are divided by
<M^4> and the cross with the full-sky input (and the monopole) by <M^2>. n0 is
already 1/f_sky normalized by Plancklens. No mean field is subtracted: biases
are measured against the alpha=None case, which shares every seed.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import healpy as hp
import numpy as np
from mpi4py import MPI

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

# Plancklens only enables MPI when $_ contains 'srun' or 'mpirun', which is
# unreliable. Point its wrapper at COMM_WORLD explicitly so that rank 0
# alone writes the shared caches and the other ranks wait at its barriers.
from plancklens.helpers import mpi as plmpi  # noqa: E402
plmpi.rank, plmpi.size = rank, size
plmpi.barrier, plmpi.bcast = comm.Barrier, comm.bcast
plmpi.send, plmpi.receive, plmpi.finalize = comm.send, comm.recv, MPI.Finalize
plmpi.ANY_SOURCE = MPI.ANY_SOURCE

from sim import CMBobs, get_lensed_cmb_spectra  # noqa: E402
from recon import AlphaRecon  # noqa: E402


def log(*args):
    print(f'[rank {rank}/{size} {time.strftime("%H:%M:%S")}]', *args, flush=True)


def parse_alpha(token):
    return None if token.lower() == 'none' else float(token)


def case_name(acb, alpha):
    return f'Acb_{acb:g}/alpha_{"None" if alpha is None else f"{alpha:g}"}'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--libdir', type=Path, required=True)
    p.add_argument('--nsims', type=int, default=100)
    p.add_argument('--first-idx', type=int, default=0)
    p.add_argument('--acb', type=float, nargs='+', default=[1e-7, 1e-6],
                   help='anisotropic amplitude(s), rad^2')
    p.add_argument('--alpha', type=parse_alpha, nargs='+',
                   default=[None, 0.3, 0.2, 0.1, 0.05, 0.001, 0.0001],
                   help="miscalibration angle(s) in degrees; 'None' for none")
    p.add_argument('--nlevp', type=float, default=2.0, help='uK-arcmin')
    p.add_argument('--fwhm', type=float, default=5.0, help='arcmin')
    p.add_argument('--nside', type=int, default=512)
    p.add_argument('--lmin-ivf', type=int, default=20)
    p.add_argument('--lmax-ivf', type=int, default=1024)
    p.add_argument('--lmax-qlm', type=int, default=1024)
    p.add_argument('--mask', type=Path, default=REPO / 'data' / 'sky2.fits')
    return p.parse_args()


def save_npz(path, **arrays):
    """Write atomically so that an interrupted job never leaves a partial file."""
    tmp = path.with_name(path.stem + f'.tmp{rank}.npz')
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def main():
    args = parse_args()
    log(f'{MPI.Get_processor_name()}: {len(os.sched_getaffinity(0))} cores, '
        f'OMP_NUM_THREADS={os.environ.get("OMP_NUM_THREADS")}')
    mask_path = args.mask.resolve()
    libdir = args.libdir.resolve()
    # sim.get_lensed_cmb_spectra caches spectra in ./data relative to the cwd.
    os.chdir(REPO)
    if rank == 0:
        get_lensed_cmb_spectra(6000)  # run CAMB once, before the other ranks read it
    comm.Barrier()

    mask = hp.read_map(str(mask_path), dtype=float)
    tag = (f'nside{args.nside}_lmin{args.lmin_ivf}_lmax{args.lmax_ivf}_qlm{args.lmax_qlm}'
           f'_nlevp{args.nlevp:g}_fwhm{args.fwhm:g}_{mask_path.stem}')
    root = libdir / tag
    cases = [(acb, alpha) for acb in args.acb for alpha in args.alpha]
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        config.update(mpi_size=size, mask=str(mask_path), cases=[case_name(*c) for c in cases])
        (root / f'config_{time.strftime("%Y%m%d-%H%M%S")}.json').write_text(json.dumps(config, indent=1))
        log(f'{len(cases)} configurations x {args.nsims} realizations -> {root}')

    # Library construction is collective: Plancklens has rank 0 write each
    # hash/filter cache (including the dense preconditioners) behind barriers.
    recs = {}
    for acb, alpha in cases:
        obs = CMBobs(nside=args.nside, nlevp=args.nlevp, fwhm=args.fwhm,
                     alpha=alpha, Acb=acb, mask=mask)
        recs[acb, alpha] = AlphaRecon(
            obs, root / 'recon' / case_name(acb, alpha), lmin_ivf=args.lmin_ivf,
            lmax_ivf=args.lmax_ivf, lmax_qlm=args.lmax_qlm, cinv=True,
            filter_libdir=root / 'filters')
        if rank == 0:
            recs[acb, alpha].response()  # fill the response cache once
    comm.Barrier()
    if rank == 0:
        log('libraries ready')

    lmax_qlm = args.lmax_qlm
    any_obs = next(iter(recs.values())).obs
    ells, ms = hp.Alm.getlm(lmax_qlm)
    selection = hp.Alm.getidx(any_obs.lmax, ells, ms)
    fsky2, fsky4 = (np.mean(any_obs.mask**n) for n in (2, 4))
    indices = range(args.first_idx, args.first_idx + args.nsims)[rank::size]
    start = time.perf_counter()
    for idx in indices:
        for acb in args.acb:
            truth = None
            for alpha in args.alpha:
                out = root / 'results' / case_name(acb, alpha) / f'sim_{idx:04d}.npz'
                if out.exists():
                    continue
                rec = recs[acb, alpha]
                if truth is None:
                    truth = rec.obs.alpha_alm(idx)[selection]
                t0 = time.perf_counter()
                est = rec.reconstruct(idx)
                out.parent.mkdir(parents=True, exist_ok=True)
                save_npz(out, auto=hp.alm2cl(est) / fsky4,
                         cross=hp.alm2cl(est, truth) / fsky2,
                         input=hp.alm2cl(truth), n0=rec.noise_spectrum(idx),
                         monopole=est[0].real / np.sqrt(4 * np.pi) / fsky2,
                         fsky2=fsky2, fsky4=fsky4)
                log(f'idx={idx} {case_name(acb, alpha)} done in '
                    f'{time.perf_counter() - t0:.0f} s (total {time.perf_counter() - start:.0f} s)')

    comm.Barrier()
    if rank == 0:
        summarize(root, args, {acb: recs[acb, args.alpha[0]].obs.cl_aa()[:lmax_qlm + 1]
                               for acb in args.acb})


def summarize(root, args, theory):
    """Stack per-realization results into one file per Acb (alpha, sim, L)."""
    indices = np.arange(args.first_idx, args.first_idx + args.nsims)
    keys = ('auto', 'cross', 'input', 'n0', 'monopole')
    for acb in args.acb:
        stacked = {k: [] for k in keys}
        for alpha in args.alpha:
            files = [root / 'results' / case_name(acb, alpha) / f'sim_{i:04d}.npz' for i in indices]
            missing = [f.name for f in files if not f.exists()]
            if missing:
                log(f'WARNING {case_name(acb, alpha)}: {len(missing)} missing, summary skipped')
                break
            data = [np.load(f) for f in files]
            for k in keys:
                stacked[k].append(np.array([d[k] for d in data]))
        else:
            out = root / f'summary_Acb_{acb:g}.npz'
            save_npz(out, ell=np.arange(args.lmax_qlm + 1), indices=indices,
                     fsky2=data[0]['fsky2'], fsky4=data[0]['fsky4'],
                     alpha=np.array([np.nan if a is None else a for a in args.alpha]),
                     theory=theory[acb],
                     **{k: np.array(v) for k, v in stacked.items()})
            log(f'wrote {out}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        comm.Abort(1)  # do not leave the other ranks waiting at a barrier
