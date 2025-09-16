import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt, pi

def f(t):
    return np.exp(-t**2)

def simpson(f, a, b, n=1000):
    if b == a:
        return 0.0
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    S = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return h * S / 3

def main():
    parser = argparse.ArgumentParser(
        description="Compute E(x)=∫_0^x e^{-t^2} dt for x=0..3 using Simpson's rule.")
    parser.add_argument(
        "-n", "--slices",
        type=int,
        default=1000,
        help="Number of Simpson subintervals (must be even; odd values will be increased by 1).")
    args = parser.parse_args()
    n = max(2, args.slices)
    if n % 2 == 1:
        n += 1
    print(f"Using n = {n} Simpson slices")

    xs = np.round(np.arange(0.0, 3.0 + 0.1, 0.1), 10)

    E_simpson = np.array([simpson(f, 0.0, x, n=n) for x in xs])

    E_ref = (sqrt(pi) / 2.0) * np.array([erf(x) for x in xs])

    print(f"{'x':>6} {'E_simpson':>14} {'E_ref':>14} {'abs_error':>14}")
    for x, Es, Er in zip(xs, E_simpson, E_ref):
        print(f"{x:6.2f} {Es:14.9f} {Er:14.9f} {abs(Es-Er):14.3e}")

    plt.plot(xs, E_simpson, label="E(x) (Simpson)")
    plt.plot(xs, E_ref, "--", label="E(x) via erf (reference)")
    plt.xlabel("x")
    plt.ylabel("E(x)")
    plt.title(r"Integral $E(x)=\int_0^x e^{-t^2}\,dt$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()