import numpy as np
import plotly.graph_objects as go

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
    x0_chk, y0_chk = 1.0, 0.5
    F = force_magnitude(M, m, L, x0_chk, y0_chk)
    ax_chk, ay_chk = accelerations(M, L, x0_chk, y0_chk)
    print(f"|F| = {F:.6f}")
    print(f"ax, ay = {ax_chk:.6f}, {ay_chk:.6f}")

G = 1.0
M = 10.0
L = 2.0

def deriv(t, ystate):
    x, y_, vx, vy = ystate
    r = np.hypot(x, y_)
    S = np.sqrt(r*r + (L/2)**2)
    ax = -G * M * x / (r**2 * S)
    ay = -G * M * y_ / (r**2 * S)
    return np.array([vx, vy, ax, ay])

def rk4_step(f, t, ystate, h):
    k1 = f(t, ystate)
    k2 = f(t + 0.5*h, ystate + 0.5*h*k1)
    k3 = f(t + 0.5*h, ystate + 0.5*h*k2)
    k4 = f(t + h,     ystate + h*k3)
    return ystate + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

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

x_traj, y_traj = Y[:, 0], Y[:, 1]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x_traj, y=y_traj,
    mode="lines",
    name="Trajectory"
))

fig.add_trace(go.Scatter(
    x=[0], y=[0],
    mode="markers",
    marker=dict(size=10),
    name="Rod axis (origin)"))

fig.update_layout(
    title="Orbit in the rod’s midplane (M=10, L=2, G=1)",
    xaxis_title="x",
    yaxis_title="y",
    width=700,
    height=700)

fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1)

fig.show()
