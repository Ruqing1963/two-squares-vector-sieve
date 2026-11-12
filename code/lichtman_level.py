"""Switched-term sieve factor with the variable level of Lichtman (arXiv:2309.08522, (6.2) and (6.4)) as used by
Runbo Li (arXiv:2405.05727, (28)-(29)) for a Chen-switched set sifted by Iwaniec's well-factorable linear sieve
weights supported on D^well, residue class N (alpha = 1, Kim-Sarnak theta = 7/32).

Upper bound for S(A'; N^{1/2}) <= S(A'; N^{mu}) by a 3-fold Buchstab expansion from z0 = N^{u}, u = 1/500 (Li (27)-(28)):
  G1 = (1/u) F(theta(u,u,u)/u) - (1/u) int_u^mu f((theta(t,u,u) - t)/u) dt/t
       + (1/u) int_u^mu int_u^{t1} F((theta(t1,t2,u) - t1 - t2)/u) dt2 dt1/(t1 t2)
       - int_u^mu int_u^{t1} int_u^{t2} f((theta(t1,t2,t3) - t1 - t2 - t3)/t3) dt3 dt2 dt1/(t1 t2 t3^2),
in units of V(N) |A'| (linear sieve, V(N^c) = V(N)/c).  With theta = 1/2 throughout the bracket must equal
2 e^gamma / (1/2) = 4 e^gamma = 7.1243 (Buchstab identity), which checks the integrator.  Li's value is G1 = 6.06932,
so the switched term is multiplied by G1 / (4 e^gamma) = 0.852.
Output: results/lichtman_level.txt"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
out = open(os.path.join(HERE, "..", "results", "lichtman_level.txt"), "w", encoding="utf8")
def say(s):
    print(s, flush=True); out.write(s + "\n")

# ---------------- linear sieve functions on a grid ----------------
g = np.euler_gamma; C = 2 * np.exp(g)
h = 1e-4
S = np.arange(h, 60 + h / 2, h); n = len(S); lag = int(round(1 / h))
F = np.where(S <= 3 + 1e-12, C / S, 0.0)
f = np.where((S >= 2 - 1e-12) & (S <= 4 + 1e-12), C * np.log(np.maximum(S - 1, 1)) / S, 0.0)
sF = S * F; sf = S * f
i3 = int(round(3 / h)) - 1; i4 = int(round(4 / h)) - 1
for i in range(i3 + 1, n):
    sF[i] = sF[i - 1] + h * 0.5 * (f[i - lag] + f[i - 1 - lag]); F[i] = sF[i] / S[i]
    if i > i4:
        sf[i] = sf[i - 1] + h * 0.5 * (F[i - lag] + F[i - 1 - lag]); f[i] = sf[i] / S[i]
def Fv(s): return F[np.clip(np.round(np.minimum(s, 59.9) / h).astype(int) - 1, 0, n - 1)]
def fv(s): return f[np.clip(np.round(np.minimum(s, 59.9) / h).astype(int) - 1, 0, n - 1)]

# ---------------- Lichtman's level functions, alpha = 1 (residue up to x^{1+eps}), theta = 7/32 ----------------
TH = 7 / 32; AL = 1.0
MU = min((1 - TH) / (4 - 3 * TH), (1 - AL * TH) / 4)          # balance point, = 25/128 for alpha = 1
def th1(t):
    t = np.asarray(t, dtype=float)
    return np.minimum(np.minimum((1 + t) / 2, (1 - (1.5 - 2 * TH) * t) / (1 + TH)), 1 - (AL * TH + 3 * t) / 2)
def psi(x, y): return np.where(x >= y, x, 0.0)
def w(t1, t2, t3):
    return psi(np.minimum(np.minimum(((5 - 4 * TH) - (3 - 4 * TH) * t3) / (8 - 6 * TH), (5 - AL * TH - 3 * t3) / 8), 1 - 2 * t2), (1 + t1) / 2)
def th3(t1, t2, t3):
    """Lichtman (6.4) for t1 <= mu; for t1 > mu only theta(t1) is available"""
    t1, t2, t3 = np.broadcast_arrays(np.asarray(t1, float), np.asarray(t2, float), np.asarray(t3, float))
    v = np.maximum.reduce([th1(t1), th1(t2), th1(t1 + t2), th1(t1 + t2 + t3), w(t1, t2, t3), w(t1, t3, t2),
                           psi(th1(t1 + t3), t1 + 2 * t2 + t3), psi(th1(t2 + t3), 2 * t1 + t2 + t3)])
    return np.where(t1 <= MU, v, th1(t1))

def bracket(level3, u=1 / 500, mu=MU, m1=4000, m2=600, m3=200):
    """G1 with the level function level3(t1,t2,t3); geometric grids in t (the integrands are ~ 1/t)"""
    t = np.geomspace(u, mu, m1)
    T0 = (1 / u) * Fv(level3(u, u, u) / u)
    T1 = (1 / u) * np.trapz(fv((level3(t, u, u) - t) / u) / t, t)
    # double integral: t2 in [u, t1]
    t1 = np.geomspace(u, mu, m2)
    I2 = np.zeros(m2)
    for k, a in enumerate(t1):
        t2 = np.geomspace(u, a, m2)
        I2[k] = np.trapz(Fv((level3(a, t2, u) - a - t2) / u) / t2, t2) / a
    T2 = (1 / u) * np.trapz(I2, t1)
    # triple integral: t3 in [u, t2], t2 in [u, t1]
    t1 = np.geomspace(u, mu, m3)
    I3 = np.zeros(m3)
    for k, a in enumerate(t1):
        t2 = np.geomspace(u, a, m3)
        J = np.zeros(m3)
        for kk, b in enumerate(t2):
            t3 = np.geomspace(u, b, m3)
            J[kk] = np.trapz(fv((level3(a, b, t3) - a - b - t3) / t3) / t3 ** 2, t3) / b
        I3[k] = np.trapz(J, t2) / a
    T3 = np.trapz(I3, t1)
    return T0 - T1 + T2 - T3, (T0, T1, T2, T3)

say("linear sieve: F(1) = %.5f, 4 e^gamma = %.5f, F(3) = %.5f, f(4) = %.5f, F(10) = %.6f" % (Fv(1.0), 2 * C, Fv(3.0), fv(4.0), Fv(10.0)))
say("Lichtman level, alpha = 1, theta = 7/32: mu = %.5f, theta(mu) = %.5f = %s, theta(u,u,u) at u = 1/500: %.6f = 19101/32000 = %.6f"
    % (MU, th1(MU), "153/256 = %.5f" % (153 / 256), th3(1 / 500, 1 / 500, 1 / 500), 19101 / 32000))
say("")
half = lambda t1, t2, t3: 0.5 * np.ones(np.broadcast(np.asarray(t1), np.asarray(t2), np.asarray(t3)).shape)
for (m1, m2, m3) in ((2000, 300, 100), (4000, 600, 200), (8000, 900, 300)):
    val, parts = bracket(half, m1=m1, m2=m2, m3=m3)
    say("check level 1/2, grids (%d,%d,%d): bracket = %.5f (should be 4 e^gamma = %.5f); parts %s" % (m1, m2, m3, val, 2 * C, " ".join("%.4f" % p for p in parts)))
say("")
for (m1, m2, m3) in ((2000, 300, 100), (4000, 600, 200), (8000, 900, 300)):
    val, parts = bracket(th3, m1=m1, m2=m2, m3=m3)
    say("Lichtman level, grids (%d,%d,%d): G1 = %.5f (Li: 6.06932); factor G1/(4 e^gamma) = %.5f; parts %s" % (m1, m2, m3, val, val / (2 * C), " ".join("%.4f" % p for p in parts)))
# Only the triple term depends on the level, so G1 = 4 e^gamma - (T3[level] - T3[1/2]) with both T3 on the same grid
# (the quadrature errors cancel).  Variants: (a) three-variable level (6.4); (b) one-variable level theta(t1) only;
# (c) (6.4) with w(t2,t1,t3) in place of w(t1,t3,t2), the form suggested by the derivation on p. 27.
say("\nsame-grid differences: G1 = 4 e^gamma - (T3[level] - T3[1/2])")
def th3b(t1, t2, t3): return th1(t1) * np.ones(np.broadcast(np.asarray(t1), np.asarray(t2), np.asarray(t3)).shape)
def th3c(t1, t2, t3):
    t1, t2, t3 = np.broadcast_arrays(np.asarray(t1, float), np.asarray(t2, float), np.asarray(t3, float))
    v = np.maximum.reduce([th1(t1), th1(t2), th1(t1 + t2), th1(t1 + t2 + t3), w(t1, t2, t3), w(t2, t1, t3),
                           psi(th1(t1 + t3), t1 + 2 * t2 + t3), psi(th1(t2 + t3), 2 * t1 + t2 + t3)])
    return np.where(t1 <= MU, v, th1(t1))
results = {}
for (m1, m2, m3) in ((4000, 600, 200), (8000, 900, 300), (8000, 900, 450)):
    _, ph = bracket(half, m1=m1, m2=m2, m3=m3)
    row = []
    for name, lev in (("(a) (6.4)", th3), ("(b) theta(t1) only", th3b), ("(c) w(t2,t1,t3) variant", th3c)):
        _, pl = bracket(lev, m1=m1, m2=m2, m3=m3)
        G1 = 2 * C - (pl[3] - ph[3]); results[name] = G1
        row.append("%s: G1 = %.4f, factor %.4f" % (name, G1, G1 / (2 * C)))
    say("  grids (%d,%d,%d): " % (m1, m2, m3) + ";  ".join(row))
G1 = results["(a) (6.4)"]
say("\nLi's value 6.06932 corresponds to the factor %.4f; variant (a) gives %.4f (effective level %.4f)" % (6.06932 / (2 * C), G1 / (2 * C), 1 / (2 * G1 / (2 * C))))

# Independent check of the triple-term difference by the midpoint rule on uniform grids (no cancellation: the difference
# f(s_L) - f(s_{1/2}) vanishes where both arguments are large, i.e. for small t3, so the integrand is bounded).
say("\nindependent check: delta T3 = int [f((theta(t1,t2,t3) - S)/t3) - f((1/2 - S)/t3)] / (t1 t2 t3^2), S = t1 + t2 + t3, midpoint rule")
def dT3(level3, m):
    u = 1 / 500; mu = MU
    t1v = u + (mu - u) * (np.arange(m) + 0.5) / m; d1 = (mu - u) / m
    tot = 0.0
    for a in t1v:
        t2v = u + (a - u) * (np.arange(m) + 0.5) / m; d2 = (a - u) / m
        T2, T3 = np.meshgrid(t2v, np.arange(m) + 0.5, indexing="ij")
        T3 = u + (T2 - u) * T3 / m; d3 = (T2 - u) / m                 # t3 in [u, t2], own cell width per t2
        S = a + T2 + T3
        diff = fv((level3(a, T2, T3) - S) / T3) - fv((0.5 - S) / T3)
        tot += np.sum(diff / (a * T2 * T3 ** 2) * d3) * d2 * d1
    return tot
for m in (100, 200, 400):
    d = dT3(th3, m)
    say("  m = %d: delta T3 = %.4f  =>  G1 = %.4f, factor %.4f" % (m, d, 2 * C - d, (2 * C - d) / (2 * C)))
# where does the difference come from: contributions by t3 ranges
say("  (Li's 6.06932 would need delta T3 = %.4f)" % (2 * C - 6.06932))
np.save(os.path.join(HERE, "..", "results", "lichtman_factor.npy"), np.array([G1 / (2 * C), 6.06932 / (2 * C)]))
