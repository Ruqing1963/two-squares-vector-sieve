"""Application 3: the problem of Nath-Xie (arXiv:2501.16723), p - 1 = m^2 + n^2 and p + 2 = P_r, with the Buchstab
fixed point on the inert primes of p - 1 (engine with kappa_aux = 1 for the linear sieve on p + 2).

Nath-Xie, Section 5:  H(lam, th, th1, th2) = A f(1/(2 th1), 1/(2 th2)) / (sqrt(th1) th2) - T(th1, th2) - lam A I(th, th1, th2) / (sqrt(th1) th2),
  A = C kappa e^{-3 gamma/2} / 4,  T = 2 c1 c2^2 c3 C(th1) / (th2 min(th1, 1/2 - th2)),
  f(s1, s2) = sup [f_{1/2}(s1) F_1(s2) + f_1(s2) F_{1/2}(s1) - F_{1/2}(s1) F_1(s2)] over th1 s1 + th2 s2 = 1/2  (= Phi_0(1/2; th2, th1)),
  I = int_{th2}^{th} (1 - al/th) F(1/(2 th1) - al/th1, 1/(2 th2) - al/th2) dal/al  (F = rectangle upper bound = G_0(1/2 - al; th2, th1)),
  C(th1) = int_0^{1 - 2 th1} log((1 - b - th1)/th1) / (sqrt(b) (1 - b)) db,  and  omega(p + 2) < 1/lam + 1/th.
Their point: H(0.17, 0.25, 0.449, 0.011) = 1.2690, 1/lam + 1/th = 9.88, so P_9.
Here f and F are replaced by the fixed point Phi, G of the Buchstab recursion on the inert primes (main set, kappa = 1/2),
the switched term T is unchanged, and we minimise 1/lam + 1/th subject to H > 0.
Output: results/nath_xie_app.txt"""
import os, sys, time
import numpy as np
sys.argv = ["x", "lib"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hjb_sieve_engine as E
out = open(os.path.join(HERE, "..", "results", "nath_xie_app.txt"), "w", encoding="utf8")
def say(s=""):
    print(s, flush=True); out.write(s + "\n"); out.flush()
T0 = time.time()
def clock(): return "[%6.0f s]" % (time.time() - T0)
g = np.euler_gamma

# ---------------- the constants (primes up to 10^7; tails are O(1/P)) ----------------
PMAX = 10 ** 7
sieve = np.ones(PMAX + 1, bool); sieve[:2] = False
for i in range(2, int(PMAX ** 0.5) + 1):
    if sieve[i]: sieve[i * i::i] = False
P = np.nonzero(sieve)[0].astype(float)
p1 = P[P % 4 == 1]; p3 = P[P % 4 == 3]
Cc = 2.25 * np.prod(1 - (3 * p3[p3 > 3] - 1) / (p3[p3 > 3] - 1) ** 3) * np.prod(1 - 1 / (p1 - 1) ** 2)
kap = np.sqrt(np.pi / 2) * np.sqrt(np.prod(1 - 1 / p3 ** 2))
c1 = np.prod(1 + 1 / (P[P > 3] - 2) ** 2); c2 = np.prod(1 - 1 / (P[P > 2] - 1) ** 2)
# c3 = (2/sqrt pi) prod_{p=1(4)} (1-1/p)^{1/2} (1 + p^2/(p-1)^3) prod_{p=3(4)} (1-1/p)^{1/2}; with prod_{p odd}(1 - chi(p)/p)^{-1} = pi/4
# this equals prod_{p=1(4)} (1 - 1/p)(1 + p^2/(p-1)^3) prod_{p=3(4)} (1 - 1/p^2)^{1/2}
c3 = np.prod((1 - 1 / p1) * (1 + p1 ** 2 / (p1 - 1) ** 3)) * np.sqrt(np.prod(1 - 1 / p3 ** 2))
A = Cc * kap * np.exp(-1.5 * g) / 4
say("constants: C = %.6f, kappa = %.6f, c1 = %.6f, c2 = %.6f, c3 = %.6f, A = C kappa e^{-3 gamma/2}/4 = %.6f" % (Cc, kap, c1, c2, c3, A))

def Cth(th1):
    bb = np.linspace(0, np.sqrt(1 - 2 * th1), 4001)[1:]        # b = bb^2 removes the b^{-1/2} singularity
    b = bb ** 2
    return np.trapz(2 * np.log((1 - b - th1) / th1) / (1 - b), bb)
def Tsw(th1, th2): return 2 * c1 * c2 ** 2 * c3 * Cth(th1) / (th2 * min(th1, 0.5 - th2))

def crit(Phi, G, n, th1, th2, th, lam):
    """H and its pieces at (th1, th2, th, lam) from fixed-point (or rectangle) tables on the grid n"""
    h = 0.5 / n; j = int(round(th1 / h)); Xg = np.arange(1, n + 1) * h
    al = np.geomspace(th2, th, 400)
    Gt = np.interp(0.5 - al, Xg, G[1:, j])
    I = np.trapz((1 - al / th) * Gt / al, al)
    Ph = Phi[n, j]
    main = A * Ph / (np.sqrt(th1) * th2); rich = A * I / (np.sqrt(th1) * th2); T = Tsw(th1, th2)
    return main - T - lam * rich, main, T, rich, Ph, I

def best_r(Phi, G, n, th1, th2):
    """minimise 1/lam + 1/th subject to H >= 0 (H is decreasing in lam): lam_max = (main - T)/rich"""
    best = None
    for th in np.linspace(th2 + 0.01, 0.6, 300):
        H0, main, T, rich, Ph, I = crit(Phi, G, n, th1, th2, th, 0.0)
        if main - T <= 0 or rich <= 0: continue
        lam = (main - T) / rich
        val = 1 / lam + 1 / th
        if best is None or val < best[0]: best = (val, th, lam, main, T, rich, Ph)
    return best

# ---------------- validation against Nath-Xie: rectangle bounds (no Buchstab) at their point ----------------
th1, th2, th, lam = 0.449, 0.011, 0.25, 0.17
n = 400
eng = E.Engine(th2, n, kappa_aux=1.0, nS=400)
P0 = np.zeros((n + 1, n + 1)); G0 = np.zeros((n + 1, n + 1))
for i in range(1, n + 1): P0[i, 1:n], G0[i, 1:n] = eng.rect_row(i)
# th1 = 0.449 is not on the grid n = 400 (h = 0.00125); use the exact rectangle at theta1 directly
def rect_exact(X, e, a, nS=2000):
    fa, Fa = E.sieve_tables(1.0); f, F = E.sieve_tables(0.5)
    smax = (X - a) / e; Pv = 0.0
    if smax > 1:
        s1 = np.geomspace(1.0, smax, nS); s3 = (X - e * s1) / a
        Pv = max(0.0, (E.lk(fa, s1) * E.lk(F, s3) + E.lk(Fa, s1) * E.lk(f, s3) - E.lk(Fa, s1) * E.lk(F, s3)).max())
    s1 = np.geomspace(1.0, X / e * (1 - 1e-6), nS); s3 = (X - e * s1) / a
    Gv = np.where(s3 > 0, E.lk(Fa, s1) * E.lk(F, np.maximum(s3, 1e-4)), np.inf).min()
    return Pv, Gv
Ph = rect_exact(0.5, th2, th1)[0]
al = np.geomspace(th2, th, 400); I = np.trapz([(1 - x / th) * rect_exact(0.5 - x, th2, th1)[1] / x for x in al], al)
main = A * Ph / (np.sqrt(th1) * th2); T = Tsw(th1, th2); rich = A * I / (np.sqrt(th1) * th2)
say("Nath-Xie point (lam, th, th1, th2) = (0.17, 0.25, 0.449, 0.011): f = %.5f, I = %.5f, C(th1) = %.5f; main %.4f, switched %.4f, lam*Richert %.4f"
    % (Ph, I, Cth(th1), main, T, lam * rich))
say("  H = %.4f (Nath-Xie: 1.2690); 1/lam + 1/th = %.4f (9.8824)  %s" % (main - T - lam * rich, 1 / lam + 1 / th, clock()))

# ---------------- search: rectangle (m = 0) and fixed point, n = 400 ----------------
say("\nminimum of 1/lam + 1/th with H >= 0 (n = 400; theta1 on the grid, step 0.0025)")
TH1 = [round(x, 4) for x in np.arange(0.30, 0.4976, 0.0025)]
for label, rule in (("rectangle (m = 0)", None), ("fixed point, trapezoid", "trapezoid"), ("fixed point, one-sided", "one-sided")):
    glob = None
    for th2 in (0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025):
        eng = E.Engine(th2, n, kappa_aux=1.0, nS=200)
        if rule is None:
            Phi = np.zeros((n + 1, n + 1)); G = np.zeros((n + 1, n + 1))
            for i in range(1, n + 1): Phi[i, 1:n], G[i, 1:n] = eng.rect_row(i)
        else:
            Phi, G = eng.solve(rule)
        for t1 in TH1:
            if t1 <= 0.25 or t1 + th2 >= 0.5: continue
            b = best_r(Phi, G, n, t1, th2)
            if b and (glob is None or b[0] < glob[0]): glob = b + (t1, th2)
    if glob is None:
        say("  %s: no (theta1, theta2) in the scan has main > switched; the inequality cannot hold  %s" % (label, clock())); continue
    val, th, lam, main, T, rich, Ph, t1, th2 = glob
    say("  %s: min 1/lam + 1/th = %.4f  => omega(p+2) <= %d   at th1 %.4f th2 %.4f th %.4f lam %.4f; main %.4f switched %.4f lam*Richert %.4f Phi %.5f  %s"
        % (label, val, int(np.ceil(val)) - 1, t1, th2, th, lam, main, T, lam * rich, Ph, clock()))
    np.save(os.path.join(HERE, "..", "results", "nath_xie_best_%s.npy" % ("rect" if rule is None else rule.replace("-", ""))), np.array([val, th, lam, t1, th2]))

# ---------------- the fixed-point optimum on finer grids ----------------
say("\nfixed point at the trapezoid optimum on n = 800 and 1600, both rules")
fn = os.path.join(HERE, "..", "results", "nath_xie_best_trapezoid.npy")
if not os.path.exists(fn): say("  (no fixed-point optimum found)"); say("done " + clock()); sys.exit(0)
val, th, lam, t1, th2 = np.load(fn)
for n in (800, 1600):
    eng = E.Engine(th2, n, kappa_aux=1.0, nS=200)
    for rule in ("trapezoid", "one-sided"):
        Phi, G = eng.solve(rule)
        b = best_r(Phi, G, n, t1, th2)
        say("  n=%d %s: min 1/lam + 1/th = %.4f => omega <= %d (th %.4f lam %.4f main %.4f switched %.4f)  %s" % (n, rule, b[0], int(np.ceil(b[0])) - 1, b[1], b[2], b[3], b[4], clock()))
say("done " + clock())
