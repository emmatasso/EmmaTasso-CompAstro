#!/usr/bin/env python3
import matplotlib
import numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

SIGMA_T=6.652e-25; C=2.99792458e10; R=6.9634e10
rng=np.random.default_rng(1)

def slab(n_e=1e20, width_km=1.0, max_steps=50000):
    L=width_km*1e5; l=1/(n_e*SIGMA_T); z=0.0; s_tot=0.0; zs=[z]; mu=1.0; k=0
    while 0<=z<=L and k<max_steps:
        s=rng.exponential(l); s_tot+=s; z2=z+mu*s
        if z2>=L: z=L; zs.append(z); break
        if z2<=0: z=0.0; zs.append(z); break
        z=z2; zs.append(z); mu=rng.uniform(-1,1); k+=1
    return np.array(zs), s_tot, (zs[-1]>=L), l, L

def ne(r): return 2.5e26*np.exp(-r/(0.096*R))
def mfp(r): return 1/(ne(r)*SIGMA_T)
def chord(r,mu): return -r*mu+np.sqrt((r*mu)**2+(R**2-r**2))

def solar_walk(max_steps=10000, keep=True):
    r=0.0; s=0.0; rs=[r] if keep else None; mu=rng.uniform(-1,1); k=0
    while r<0.9*R and k<max_steps:
        step=rng.exponential(mfp(r)); r2=r+mu*step
        if r2<0: s+=r/(-mu); step-=r/(-mu); r=0.0; mu=abs(mu); r+=mu*step; s+=step
        else: r=r2; s+=step
        if r<0.9*R:
            mu=rng.uniform(-1,1); k+=1; 
            if keep: rs.append(r)
    if r>=0.9*R:
        s+=chord(r,mu); 
        if keep: rs.append(R)
    return s/C, s, (np.array(rs) if keep else None), k

def main():
    zs, s, esc, l, L = slab()
    plt.figure(figsize=(6,3))
    plt.plot(np.arange(len(zs)), zs)
    plt.xlabel("Step"); plt.ylabel("z (cm)"); plt.title("1 km slab")
    plt.tight_layout(); plt.savefig("slab_trajectory.png", dpi=150)

    fig_gif, ax = plt.subplots(figsize=(6,3))
    ax.set_xlim(0, len(zs)-1); ax.set_ylim(0, L)
    ax.set_xlabel("Step"); ax.set_ylabel("z (cm)"); ax.set_title("Slab walk")
    line, = ax.plot([], [])
    def up(i):
        x = np.arange(i+1); line.set_data(x, zs[:i+1]); return (line,)
    FuncAnimation(fig_gif, up, frames=len(zs), interval=60, blit=True)\
        .save("slab_walk.gif", writer=PillowWriter(fps=15))

    t, _, rs, _ = solar_walk(keep=True)
    plt.figure(figsize=(5,4))
    plt.plot(np.arange(len(rs)), rs)
    plt.xlabel("Scatter index"); plt.ylabel("r (cm)")
    plt.title("Solar random walk (one)")
    plt.tight_layout(); plt.savefig("solar_trajectory.png", dpi=150)

    N = 2000
    times = np.empty(N); scat = np.empty(N, int)
    for i in range(N):
        ti, _, _, ki = solar_walk(keep=False)   
        times[i] = ti; scat[i] = ki
    plt.figure(figsize=(5,4))
    plt.hist(times, bins=40)
    plt.xlabel("Escape time t (s)"); plt.ylabel("Count")
    plt.title("Solar escape times")
    plt.tight_layout(); plt.savefig("solar_escape_times.png", dpi=150)

    print(f"Slab: l={l:.3e} cm, L={L:.3e} cm, escaped={esc}")
    print(f"Solar: mean t={times.mean():.3e} s, median t={np.median(times):.3e} s, "
          f"mean scatters={scat.mean():.1f}")

    plt.show()

if __name__=="__main__": main()
