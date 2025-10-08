import argparse, numpy as np, matplotlib.pyplot as plt
from astropy.io import fits

DEFAULT_FITS = "tic0000627436.fits"

def load_lightcurve(path):
    with fits.open(path) as h:
        d = h[1].data
        cols = {c.upper(): c for c in d.columns.names}
        tk = 'times' if 'times' in d.columns.names else (cols.get('TIME') or 'TIME')
        fk = 'fluxes' if 'fluxes' in d.columns.names else (cols.get('PDCSAP_FLUX') or 'PDCSAP_FLUX')
        t, f = np.array(d[tk], float), np.array(d[fk], float)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    s = np.argsort(t); t, f = t[s], f[s]
    return t, f/np.median(f) - 1.0

def dense_centers(t, w):
    centers = np.linspace(t.min()+w/2, t.max()-w/2, 500)
    counts = np.array([np.sum((t>=c-w/2)&(t<=c+w/2)) for c in centers])
    i1 = counts.argmax()
    mask = np.abs(centers - centers[i1]) > 0.8*w
    i2 = (counts*mask).argmax()
    return [centers[i1], centers[i2]]

def analyze_epoch(tt, ff, k, topN):
    dts = np.diff(tt); dtm = np.median(dts); gaps = np.sum(dts > 1.5*dtm)
    print(f"[epoch {k}] N={len(tt)}, median Δt={dtm:.6f} d, gaps>{1.5:.1f}×Δt: {gaps}")

    te = np.arange(tt.min(), tt.max()+0.5*dtm, dtm)
    fe = np.interp(te, tt, ff)

    y0 = fe - fe.mean()
    F = np.fft.rfft(y0)
    freq = np.fft.rfftfreq(len(fe), d=dtm)
    power = np.abs(F)**2

    keep = np.zeros_like(F)
    idx = np.argsort(np.abs(F[1:]))[::-1][:topN] + 1
    keep[idx] = F[idx]
    rec = np.fft.irfft(keep, n=len(fe)) + fe.mean()

    plt.figure(figsize=(9,3))
    plt.plot(tt, ff, lw=0.7)
    plt.xlabel("Time [d]"); plt.ylabel("rel flux")
    plt.title(f"Light curve (epoch {k})")
    plt.tight_layout()

    plt.figure(figsize=(9,3))
    plt.plot(freq, power, lw=0.8)
    plt.xlabel("Freq [1/d]"); plt.ylabel("Power")
    plt.title(f"FFT Power (filled, epoch {k})")
    plt.tight_layout()

    plt.figure(figsize=(9,3))
    plt.plot(te, fe, lw=0.7, label="original (filled)")
    plt.plot(te, rec, lw=1.1, label=f"top {topN} modes")
    plt.xlabel("Time [d]"); plt.ylabel("rel flux")
    plt.title(f"Sparse reconstruction (epoch {k})")
    plt.legend(); plt.tight_layout()

def main():
    parser = argparse.ArgumentParser(description="FFT analysis.")
    parser.add_argument("--fits", default=DEFAULT_FITS, help=f"FITS path (default: {DEFAULT_FITS})")
    parser.add_argument("-w", "--window", type=float, default=10.0, help="Window length in days")
    parser.add_argument("-N", "--modes", type=int, default=20, help="Top Fourier modes to keep")
    args = parser.parse_args()

    t, f = load_lightcurve(args.fits)
    centers = dense_centers(t, args.window)

    
    for k, c in enumerate(centers, 1):
        tmin, tmax = c - args.window/2, c + args.window/2
        sel = (t>=tmin) & (t<=tmax)
        tt, ff = t[sel], f[sel]
        if len(tt) < 10:
            print(f"[epoch {k}] too few points, skipping")
            continue
        print(f"[epoch {k}] range {tmin:.3f}-{tmax:.3f} d")
        analyze_epoch(tt, ff, k, args.modes)

    plt.show()

if __name__ == "__main__":
    main()
