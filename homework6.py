import numpy as np
import matplotlib.pyplot as plt

G = 1.0  

def r_perp(x, y):
    """Perpendicular distance from the rod's midpoint (lying along the z–axis)."""
    return np.hypot(x, y)

def force_magnitude(M, m, L, x, y, G=G):
    """
    |F| toward the rod's center for a rod of mass M and length L
    and a point mass m located at (x,y) in the midplane.
    """
    r = r_perp(x, y)
    S = np.sqrt(r**2 + (L/2)**2)
    return G * M * m / (r * S)

def accelerations(M, L, x, y, G=G):
    """
    Components of acceleration (d2x/dt2, d2y/dt2) for the ball bearing.
    """
    r = r_perp(x, y)
    S = np.sqrt(r**2 + (L/2)**2)
    ax = -G * M * x / (r**2 * S)
    ay = -G * M * y / (r**2 * S)
    return ax, ay

if __name__ == "__main__":
    M, m, L = 10.0, 1.0, 2.0
    x, y = 1.0, 0.5
    F = force_magnitude(M, m, L, x, y)
    ax, ay = accelerations(M, L, x, y)
    print(f"|F| = {F:.6f}")
    print(f"ax, ay = {ax:.6f}, {ay:.6f}")

G = 1.0
M = 10.0
L = 2.0

def deriv(t, y):
    x, y_, vx, vy = y
    r = np.hypot(x, y_)                     
    S = np.sqrt(r*r + (L/2)**2)
    ax = -G * M * x / (r**2 * S)
    ay = -G * M * y_ / (r**2 * S)
    return np.array([vx, vy, ax, ay])

def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h,     y + h*k3)
    return y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Initial conditions: (x,y)=(1,0), velocity = +1 in y-direction → vx=0, vy=+1
y0 = np.array([1.0, 0.0, 0.0, 1.0])

t0, tf = 0.0, 10.0
N = 20000                        
h = (tf - t0) / N

ts = np.linspace(t0, tf, N+1)
Y  = np.zeros((N+1, 4))
Y[0] = y0

t = t0
for n in range(N):
    Y[n+1] = rk4_step(deriv, t, Y[n], h)
    t += h

x, y = Y[:,0], Y[:,1]

plt.figure(figsize=(6,6))
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Orbit in the rod’s midplane (M=10, L=2, G=1)")
plt.axis("equal")
plt.tight_layout()
plt.show()