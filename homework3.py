G = 6.67430e-11      # m^3 kg^-1 s^-2
M = 5.97219e24       # kg
m = 7.342e22         # kg
R = 3.844e8          # m
w = 2.662e-6         # s^-1

def f(r):
    return G*M/r**2 - G*m/(R - r)**2 - (w**2)*r

# Secant method
x0, x1 = 0.70*R, 0.90*R
tol = 1e-12
for _ in range(100):
    fx0, fx1 = f(x0), f(x1)
    x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
    if abs(x2 - x1) <= tol*max(1.0, abs(x2)):
        r = x2
        break
    x0, x1 = x1, x2

km = 1e-3
print(f"r (from Earth) = {r:.6f} m = {r*km:.3f} km")
print(f"R - r (to Moon) = {(R - r)*km:.3f} km")
