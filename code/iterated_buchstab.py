"""Iterated Buchstab steps on the inert primes (m = 1, 2, 3, 4) by 2-D dynamic programming.

Lemma 4.3 applied m times.  For level exponent X in (0, 1/2], split primes < N^eta, inert primes < N^a:
  Phi_m(X; a) = max_{0 < theta0 <= a} [ sqrt(a/theta0) Phi_0(X; theta0) - 1/2 int_{theta0}^{a} sqrt(a/b) G_{m-1}(X - b; b) db/b ]
  G_m(X; a)   = min_{0 < theta0 <= a} [ sqrt(a/theta0) G_0(X; theta0)   - 1/2 int_{theta0}^{a} sqrt(a/b) Phi_{m-1}(X - b; b) db/b ]
with Phi_0, G_0 the rectangle (vector sieve) bounds of Section 4.7.  Phi_m is a lower bound and G_m an upper
bound for the normalised sifted count of every sub-problem (Buchstab's identity is exact; each sub-problem has
modulus <= N^{1/2-eps}; a modulus k occurs at most tau(k)^{m+1} times, which Theorem 1.1 absorbs).
Taking theta0 = a gives the previous step, so Phi_m >= Phi_{m-1} and G_m <= G_{m-1}; we enforce this.
Where the level X - b <= 0 no upper bound exists, so cuts with a >= X are not used in Phi_m; the lower bound
of a sub-problem without room is S >= 0.

Grid: X = i h, a = j h, h = 0.5/n, so that X - b lies on the grid (no interpolation in X).  The b-integral uses the
exact measure of b^{-3/2} on each cell:  1/2 int_{b_l}^{b_{l+1}} sqrt(a/b) H db/b ~ sqrt(a) (H_l + H_{l+1})/2 (b_l^{-1/2} - b_{l+1}^{-1/2}).

Checks: (i) eta -> 0: Phi_m -> f(X/a), G_m -> F(X/a) for every m (the gain of m >= 2 over m = 1 must vanish);
(ii) m = 1 reproduces Table 1; (iii) margins M_m at the optima and after retuning; (iv) grid refinement n = 200, 400, 800.
Output: results/iterated_buchstab.txt.   Run time about 25 minutes."""
import os, sys, time, importlib.util
import numpy as np
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"      # "fine": only part (viii), n up to 1600 (about 40 minutes)
sys.argv = ["x"]
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ss", os.path.join(HERE, "semilinear_sieve.py"))
ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
out = open(os.path.join(HERE, "..", "results", "iterated_buchstab%s.txt" % ("_" + MODE if MODE != "all" else "")), "w", encoding="utf8")
def say(s=""):
    print(s, flush=True); out.write(s + "\n"); out.flush()

f12, F12, lk, c = ss.f12, ss.F12, ss.lk, ss.c
T0 = time.time()
def clock(): return "[%5.0f s]" % (time.time() - T0)

def tables0(eta, n, nS1=400, nS3=600):
    """rectangle bounds Phi_0[i, j], G_0[i, j] at X = i h, a = j h (i = 1..n, j = 1..n-1); index 0 unused"""
    h = 0.5 / n
    P0 = np.zeros((n + 1, n + 1)); G0 = np.full((n + 1, n + 1), np.inf)
    a = np.arange(1, n) * h
    u = np.linspace(0.0, 1.0, nS1)
    for i in range(1, n + 1):
        X = i * h
        smax = (X - a) / eta                       # s1 in [1, smax] geometric, s3 = (X - eta s1)/a >= 1
        ok = np.nonzero(smax > 1.0)[0]
        if len(ok):
            s1 = smax[ok, None] ** u[None, :]
            s3 = (X - eta * s1) / a[ok, None]
            f1, F1, f3, F3 = lk(f12, s1), lk(F12, s1), lk(f12, s3), lk(F12, s3)
            P0[i, ok + 1] = np.maximum(0.0, (f1 * F3 + F1 * f3 - F1 * F3).max(axis=1))
        s1 = np.geomspace(1e-4, X / eta * (1 - 1e-6), nS3)
        F1 = lk(F12, s1)
        s3 = (X - eta * s1[None, :]) / a[:, None]
        F3 = np.where(s3 >= 1e-4, lk(F12, s3), np.inf)
        G0[i, 1:n] = (F1[None, :] * F3).min(axis=1)
    return P0, G0

def step(P0, G0, Pp, Gp, n, TRI, safe=False):
    """(Phi_m, G_m) from (Phi_{m-1}, G_{m-1}) = (Pp, Gp).
    safe=True: one-sided cell rule (max of the endpoint values in the term subtracted from Phi_m, min in the term
    subtracted from G_m), which is rigorous whenever the integrand is monotone on each cell."""
    h = 0.5 / n
    cP = (lambda A, B: 2 * np.maximum(A, B)) if safe else (lambda A, B: A + B)
    cG = (lambda A, B: 2 * np.minimum(A, B)) if safe else (lambda A, B: A + B)
    a = np.arange(0, n + 1) * h
    ra = np.zeros(n + 1); ra[1:] = a[1:] ** -0.5
    w = np.zeros(n + 1); w[1:n] = 0.5 * (ra[1:n] - ra[2:n + 1])      # cell [a_l, a_{l+1}] weight (times H_l + H_{l+1})
    sa = np.sqrt(a)
    Pn = Pp.copy(); Gn = Gp.copy()
    for i in range(1, n + 1):
        if i >= 3:                                   # lower bound: cuts and a up to a_{i-1} = X - h
            J = i - 1
            l = np.arange(1, J + 1)
            H = Gp[i - l, l]                         # G_{m-1}(X - a_l; a_l)
            K = np.concatenate([[0.0], np.cumsum(w[1:J] * cP(H[:-1], H[1:]))])   # K_j = int_{a_1}^{a_j}
            base = P0[i, 1:J + 1] * ra[1:J + 1] + K
            M = np.where(TRI[:J, :J], base[None, :] - K[:, None], -np.inf)
            Pn[i, 1:J + 1] = np.maximum(Pp[i, 1:J + 1], sa[1:J + 1] * M.max(axis=1))
        l = np.arange(1, n)                          # upper bound: all a_j, j = 1..n-1
        Hp = np.zeros(n - 1)
        ll = l[l < i]
        Hp[:len(ll)] = Pp[i - ll, ll]                # Phi^+_{m-1}(X - a_l; a_l), 0 without room
        K = np.concatenate([[0.0], np.cumsum(w[1:n - 1] * cG(Hp[:-1], Hp[1:]))])
        base = G0[i, 1:n] * ra[1:n] + K
        M = np.where(TRI[:n - 1, :n - 1], base[None, :] - K[:, None], np.inf)
        Gn[i, 1:n] = np.minimum(Gp[i, 1:n], sa[1:n] * M.min(axis=1))
    return Pn, Gn

def all_tables(eta, n, mmax=4, safe=False):
    TRI = np.tri(n, dtype=bool)
    P0, G0 = tables0(eta, n)
    tabs = [(P0, G0)]
    for m in range(1, mmax + 1):
        tabs.append(step(P0, G0, tabs[-1][0], tabs[-1][1], n, TRI, safe))
    return tabs

JP = {}
def jparts(eta, th):
    hw = 5e-4 if eta < 0.006 else 1e-3
    key = (eta, round(th, 5), hw)
    if key not in JP: JP[key] = ss.Jparts(eta, th, hw)
    return JP[key]

def margin(r, L, eta, th, P, G, n, nT=240, nu=300):
    """best weighted margin over u0 at (eta, th) with the tables (P, G); returns dict"""
    h = 0.5 / n; j = int(round(th / h))
    assert abs(j * h - th) < 1e-9, "theta must lie on the grid"
    Xg = np.arange(1, n + 1) * h
    ts = np.geomspace(eta, 0.49, nT)
    Gt = np.interp(0.5 - ts, Xg, G[1:, j])
    Ph = P[n, j]
    parts = jparts(eta, th)
    j0 = ss.Jlam(parts, 0.0, 1.0); best = None
    for u0 in np.linspace(eta + 0.01, 0.48, nu):
        il = r + 1 - 1e-6 - 1 / u0
        if il <= 0: continue
        lam = 1 / il; k = ts <= u0
        if k.sum() < 2: continue
        integ = np.trapz((1 - ts[k] / u0) * Gt[k] / ts[k], ts[k])
        main = c * (Ph - lam / 2 * integ) / np.sqrt(eta * th)
        jj = ss.Jlam(parts, lam, u0) if 1 - 2 * th <= u0 else j0
        M = main - 8 * L * jj
        if best is None or M > best["M"]:
            best = dict(M=M, main=main, T=8 * L * jj, u0=u0, ilam=1 / lam, Phi=Ph, integ=integ)
    return best

def fmt(b):
    return "M %+.4f (%+5.1f%%)  main %.4f  sw %.4f  Phi %.5f  Rint %.4f  u0 %.3f  1/lam %.2f" % (
        b["M"], 100 * b["M"] / b["main"], b["main"], b["T"], b["Phi"], b["integ"], b["u0"], b["ilam"])

# ------------------------------------------------------------------ (viii) fine grids at the fixed point (MODE == "fine")
if MODE == "fine":
    say("(viii) fixed point (m = 12) on n = 400, 800, 1600 at the retuned parameters of part (vii) " + clock())
    MM = 12
    for (L, rs, eta, th) in [(1.0, (6, 7), 0.015, 0.370), (1.5, (9, 10), 0.010, 0.420), (3.0, (15, 16), 0.005, 0.460)]:
        for nn in (400, 800, 1600):
            tabs = all_tables(eta, nn, MM)
            for r in rs:
                b = [margin(r, L, eta, th, tabs[m][0], tabs[m][1], nn) for m in (1, 2, 4, 8, 12)]
                say("  L=%.1f r=%d eta=%.4f th=%.3f n=%d: M_1 %+.4f  M_2 %+.4f  M_4 %+.4f  M_8 %+.4f  M_12 %+.4f   (m=12: %s)  %s" % (
                    L, r, eta, th, nn, b[0]["M"], b[1]["M"], b[2]["M"], b[3]["M"], b[4]["M"], fmt(b[4]), clock()))
            del tabs
    # eta -> 0 error of the fixed point on the finest grid
    nn = 1600; h = 0.5 / nn
    tabs = all_tables(1e-7, nn, MM)
    say("  eta = 1e-7, n = 1600, m = 12: Phi_12 - f at (1/2, 0.37) %+.5f, (1/2, 0.42) %+.5f, (1/2, 0.46) %+.5f; G_12 - F at (0.15, 0.35) %+.5f  %s" % (
        tabs[MM][0][nn, int(round(0.37 / h))] - ss.ff(0.5 / 0.37)[0], tabs[MM][0][nn, int(round(0.42 / h))] - ss.ff(0.5 / 0.42)[0],
        tabs[MM][0][nn, int(round(0.46 / h))] - ss.ff(0.5 / 0.46)[0], tabs[MM][1][int(round(0.15 / h)), int(round(0.35 / h))] - ss.ff(0.15 / 0.35)[1], clock()))
    say("done " + clock()); sys.exit(0)

# ------------------------------------------------------------------ (ix) one-sided cell rule (MODE == "safe")
if MODE == "safe":
    say("(ix) one-sided cell rule (safe=True) against the trapezoid rule, m = 12, n = 400 and 800 " + clock())
    MM = 12
    for nn in (400, 800):
        h = 0.5 / nn
        tabs = all_tables(1e-7, nn, MM, safe=True)
        say("  eta = 1e-7, n = %d, safe: Phi_12 - f at (1/2, 0.37) %+.5f, (1/2, 0.42) %+.5f, (1/2, 0.46) %+.5f; G_12 - F at (0.15, 0.35) %+.5f" % (nn,
            tabs[MM][0][nn, int(round(0.37 / h))] - ss.ff(0.5 / 0.37)[0], tabs[MM][0][nn, int(round(0.42 / h))] - ss.ff(0.5 / 0.42)[0],
            tabs[MM][0][nn, int(round(0.46 / h))] - ss.ff(0.5 / 0.46)[0], tabs[MM][1][int(round(0.15 / h)), int(round(0.35 / h))] - ss.ff(0.15 / 0.35)[1]))
    for (L, rs, eta, th) in [(1.0, (6, 7), 0.015, 0.370), (1.5, (9, 10), 0.010, 0.420), (3.0, (15, 16), 0.005, 0.460)]:
        for nn in (400, 800):
            for safe in (False, True):
                tabs = all_tables(eta, nn, MM, safe)
                for r in rs:
                    b = margin(r, L, eta, th, tabs[MM][0], tabs[MM][1], nn)
                    say("  L=%.1f r=%d eta=%.4f th=%.3f n=%d %s: M_12 %+.4f  Phi %.5f  %s" % (L, r, eta, th, nn, "safe " if safe else "trap ", b["M"], b["Phi"], clock()))
    say("done " + clock()); sys.exit(0)

if MODE == "curve":
    # margin of the fixed point (m = 12) against r, retuned over eta and theta (n = 400), for Figure 1;
    # and the rectangle values m = 0 at the parameters of Table 1 on n = 1600
    say("(xi) fixed point (m = 12) against r, n = 400, retuned over eta and theta " + clock())
    setups = {1.0: ([0.006, 0.008, 0.01, 0.012, 0.015, 0.02], range(6, 12)),
              1.5: ([0.004, 0.005, 0.006, 0.008, 0.01, 0.012], range(9, 15)),
              3.0: ([0.002, 0.0025, 0.003, 0.004, 0.005, 0.006], range(15, 22))}
    ths = [round(x, 4) for x in np.arange(0.33, 0.4951, 0.005)]
    n = 400; MM = 12; curves = {}
    for L, (etas, rs) in setups.items():
        best = {r: None for r in rs}
        for eta in etas:
            tabs = all_tables(eta, n, MM); P, G = tabs[MM]
            for th in ths:
                for r in rs:
                    b = margin(r, L, eta, th, P, G, n)
                    if best[r] is None or b["M"] > best[r]["M"]: best[r] = dict(b, eta=eta, th=th)
            say("  L=%.1f eta=%.4f done %s" % (L, eta, clock()))
        curves[str(L)] = [(r, b["M"], b["main"], b["T"], b["eta"], b["th"], b["u0"], b["ilam"]) for r, b in best.items()]
        for r, b in best.items():
            say("  L=%.1f r=%2d  eta %.4f th %.3f  %s" % (L, r, b["eta"], b["th"], fmt(b)))
        say("  => L=%.1f: least r with M > 0 on this grid (trapezoid, n = 400): %d; the certified r_0 uses the one-sided rule on n = 1600 (Table 1)"
            % (L, min(r for r, b in best.items() if b["M"] > 0)))
    np.save(os.path.join(HERE, "..", "results", "iterated_buchstab_curves.npy"), curves, allow_pickle=True)
    say("\n  rectangle bounds (m = 0) at the parameters of Table 1, n = 1600:")
    for (L, r, eta, th) in [(1.0, 7, 0.015, 0.370), (1.5, 10, 0.010, 0.420), (3.0, 16, 0.005, 0.460)]:
        P0, G0 = tables0(eta, 1600)
        say("  L=%.1f r=%d eta=%.4f th=%.3f n=1600 m=0: %s  %s" % (L, r, eta, th, fmt(margin(r, L, eta, th, P0, G0, 1600)), clock()))
    say("done " + clock()); sys.exit(0)

if MODE == "retune":
    # probe A2: fine retuning of (eta, theta, u0) for r = 9 (L = 3/2) and r = 15 (L = 3) at the fixed point;
    # eta step 0.001, theta step 0.0025 (on the grids n = 400, 800, 1600); then refinement and Richardson extrapolation
    say("(xii) fine retuning for r = 9 (L = 3/2) and r = 15 (L = 3), m = 12 " + clock())
    MM = 12
    for (L, r, etas, ths) in [(1.5, 9, [round(0.007 + 0.001 * i, 3) for i in range(7)], [round(0.4125 + 0.0025 * i, 4) for i in range(9)]),
                              (3.0, 15, [round(0.003 + 0.001 * i, 3) for i in range(5)], [round(0.4525 + 0.0025 * i, 4) for i in range(9)])]:
        allb = []
        for eta in etas:
            tabs = all_tables(eta, 400, MM); P, G = tabs[MM]
            for th in ths:
                b = margin(r, L, eta, th, P, G, 400); allb.append(dict(b, eta=eta, th=th))
            say("  L=%.1f r=%d eta=%.3f: best over theta %+.4f at theta %.4f  %s" % (L, r, eta, max(x["M"] for x in allb if x["eta"] == eta),
                max((x for x in allb if x["eta"] == eta), key=lambda x: x["M"])["th"], clock()))
        allb.sort(key=lambda x: -x["M"])
        say("  top 3 on n = 400 (trapezoid):")
        for b in allb[:3]: say("    eta %.3f th %.4f  %s" % (b["eta"], b["th"], fmt(b)))
        for i, b in enumerate(allb[:3]):
            eta, th = b["eta"], b["th"]
            vals = {}
            for nn in (400, 800, 1600):
                if nn == 1600 and i > 0: continue
                for safe in (False, True):
                    if nn == 1600 and safe and i > 0: continue
                    tabs = all_tables(eta, nn, MM, safe)
                    bb = margin(r, L, eta, th, tabs[MM][0], tabs[MM][1], nn); vals[(nn, safe)] = bb["M"]
                    say("    eta %.3f th %.4f n=%d %s: M_12 %+.4f  (%s)  %s" % (eta, th, nn, "safe" if safe else "trap", bb["M"], fmt(bb), clock()))
                    del tabs
            for safe in (False, True):
                v = [vals.get((nn, safe)) for nn in (400, 800, 1600)]
                if None in v: continue
                d1, d2 = v[1] - v[0], v[2] - v[1]
                if d1 * d2 > 0 and abs(d2) < abs(d1):
                    p = np.log2(d1 / d2); lim = v[2] + d2 / (2 ** p - 1)
                    say("    Richardson (%s): differences %+.4f, %+.4f, order p = %.2f, extrapolated limit %+.4f; with p = 1: %+.4f" % (
                        "safe" if safe else "trap", d1, d2, p, lim, v[2] + d2))
                else:
                    say("    Richardson (%s): differences %+.4f, %+.4f, not geometric" % ("safe" if safe else "trap", d1, d2))
    say("done " + clock()); sys.exit(0)

if MODE == "whatif":
    # probe A1 (what-if): the switched term at level N^theta_sw instead of N^{1/2} has the constant F_lin(2 theta_sw) = 2 e^gamma / (2 theta_sw),
    # so M(theta_sw) = main - 8 L J_lambda / (2 theta_sw).  Least r with M > 0 (fixed point m = 12, n = 400, retuned) for several theta_sw.
    say("(xiii) what-if levels theta_sw for the switched term; fixed point m = 12, n = 400, retuned over eta, theta, u0 " + clock())
    THS = [0.5, 0.51, 0.525, 0.55, 0.597]
    setups = {1.0: ([0.006, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025], range(4, 10)),
              1.5: ([0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015], range(6, 13)),
              3.0: ([0.002, 0.0025, 0.003, 0.004, 0.005, 0.006, 0.008], range(11, 19))}
    ths = [round(x, 4) for x in np.arange(0.33, 0.4951, 0.005)]
    n = 400; MM = 12; h = 0.5 / n
    for L, (etas, rs) in setups.items():
        best = {}
        for eta in etas:
            tabs = all_tables(eta, n, MM); P, G = tabs[MM]
            Xg = np.arange(1, n + 1) * h; ts = np.geomspace(eta, 0.49, 240)
            for th in ths:
                j = int(round(th / h)); Gt = np.interp(0.5 - ts, Xg, G[1:, j]); Ph = P[n, j]
                parts = jparts(eta, th); j0 = ss.Jlam(parts, 0.0, 1.0)
                for r in rs:
                    for u0 in np.linspace(eta + 0.01, 0.48, 300):
                        il = r + 1 - 1e-6 - 1 / u0
                        if il <= 0: continue
                        lam = 1 / il; k = ts <= u0
                        if k.sum() < 2: continue
                        integ = np.trapz((1 - ts[k] / u0) * Gt[k] / ts[k], ts[k])
                        main = c * (Ph - lam / 2 * integ) / np.sqrt(eta * th)
                        T = 8 * L * (ss.Jlam(parts, lam, u0) if 1 - 2 * th <= u0 else j0)
                        for tsw in THS:
                            M = main - T / (2 * tsw)
                            if (r, tsw) not in best or M > best[(r, tsw)][0]: best[(r, tsw)] = (M, main, T, eta, th, u0)
            say("  L=%.1f eta=%.4f done %s" % (L, eta, clock()))
        say("  L=%.1f: margin M(theta_sw) (trapezoid, n = 400); rows r, columns theta_sw = %s" % (L, THS))
        for r in rs:
            say("    r=%2d: " % r + "  ".join("%+.3f" % best[(r, t)][0] for t in THS) + "   (theta_sw = 1/2 optimum: eta %.4f th %.3f u0 %.3f)" % best[(r, 0.5)][3:6])
        say("    least r with M > 0: " + ", ".join("theta_sw %.3f -> r_0 = %s" % (t, min([r for r in rs if best[(r, t)][0] > 0], default="none")) for t in THS))
    say("done " + clock()); sys.exit(0)

if MODE == "safe1600":
    say("(x) one-sided cell rule at n = 1600, m = 12 " + clock())
    for (L, rs, eta, th) in [(3.0, (15, 16), 0.005, 0.460), (1.0, (6, 7), 0.015, 0.370), (1.5, (9, 10), 0.010, 0.420)]:
        tabs = all_tables(eta, 1600, 12, safe=True)
        for r in rs:
            b = margin(r, L, eta, th, tabs[12][0], tabs[12][1], 1600)
            say("  L=%.1f r=%d eta=%.4f th=%.3f n=1600 safe: M_12 %+.4f  Phi %.5f  %s  %s" % (L, r, eta, th, b["M"], b["Phi"], fmt(b), clock()))
        del tabs
    say("done " + clock()); sys.exit(0)

# ------------------------------------------------------------------ (i) eta -> 0
say("Iterated Buchstab steps, 2-D dynamic programming " + clock())
say("\n(i) eta -> 0 self-check, eta = 1e-7, n = 200 (h = 0.0025): Phi_m vs f(X/a), G_m vs F(X/a)")
n = 200; h = 0.5 / n
tabs = all_tables(1e-7, n)
for (X, a) in [(0.5, 0.37), (0.5, 0.42), (0.5, 0.465), (0.5, 0.25), (0.3, 0.1), (0.15, 0.05), (0.15, 0.35)]:
    i, j = int(round(X / h)), int(round(a / h))
    fx, Fx = ss.ff(X / a)
    say("  X=%.3f a=%.3f  f=%.5f  Phi_m: %s   F=%.5f  G_m: %s" % (X, a, fx,
        " ".join("%.5f" % tabs[m][0][i, j] for m in range(5)), Fx, " ".join("%.5f" % tabs[m][1][i, j] for m in range(5))))
# global deviation on the region a <= X - 0.05 (room for the sieve), X >= 0.1; the unsafe sides are Phi_m > f and G_m < F
say("  discretisation error of the DP (eta = 1e-7) under refinement; region a <= X - 0.05, X >= 0.1:")
for n in (100, 200, 400):
    h = 0.5 / n
    tabs = all_tables(1e-7, n)
    ii, jj = np.meshgrid(np.arange(1, n + 1), np.arange(1, n), indexing="ij")
    reg = (jj * h <= ii * h - 0.05) & (ii * h >= 0.1)
    fE = np.array([ss.ff(s)[0] for s in (ii * h / (jj * h))[reg]]); FE = np.array([ss.ff(s)[1] for s in (ii * h / (jj * h))[reg]])
    for m in range(5):
        dP = tabs[m][0][1:, 1:n][reg] - fE; dG = tabs[m][1][1:, 1:n][reg] - FE
        say("   n=%d m=%d  Phi_m - f in [%+.5f, %+.5f]  G_m - F in [%+.5f, %+.5f]  at (1/2, 0.37): Phi %+.5f  (1/2, 0.465): Phi %+.5f" % (
            n, m, dP.min(), dP.max(), dG.min(), dG.max(), tabs[m][0][n, int(round(0.37 / h))] - ss.ff(0.5 / 0.37)[0],
            tabs[m][0][n, int(round(0.465 / h))] - ss.ff(0.5 / 0.465)[0]))
say(clock())

# ------------------------------------------------------------------ (ii) the example of the proposal, and m = 1 against Table 1
say("\n(ii) eta = 0.008, n = 400: the sub-problem G(0.15; 0.008, 0.35) of the proposal")
n = 400; h = 0.5 / n
tabs = all_tables(0.008, n)
i, j = int(round(0.15 / h)), int(round(0.35 / h))
say("  G_0 = %.5f (GX direct %.5f),  G_1 = %.5f,  G_2 = %.5f,  G_3 = %.5f" % (
    tabs[0][1][i, j], ss.GX(0.15, 0.008, 0.35), tabs[1][1][i, j], tabs[2][1][i, j], tabs[3][1][i, j]))
# bracket of G_1 as a function of the cut
a = np.arange(0, n + 1) * h; ra = np.zeros(n + 1); ra[1:] = a[1:] ** -0.5
l = np.arange(1, j + 1); Hp = np.where(l < i, tabs[0][0][np.minimum(i - l, n), l], 0.0)
K = np.concatenate([[0.0], np.cumsum(0.5 * (ra[1:j] - ra[2:j + 1]) * (Hp[:-1] + Hp[1:]))])
br = np.sqrt(a[j]) * (tabs[0][1][i, 1:j + 1] * ra[1:j + 1] + K - K[-1])
kb = int(np.argmin(br))
say("  bracket at cut 0.03: %.5f;  best cut %.4f gives %.5f;  (proposal: 2.50681 -> 2.48383 at cut 0.03)" % (br[int(round(0.03 / h)) - 1], a[kb + 1], br[kb]))

say("\n(iii) margins M_m, m = 0..4, at the parameters of Table 1 and one r lower (n = 400, h = 0.00125, production hw)")
CASES = [(1.0, 0.008, 0.370, (7, 8)), (1.5, 0.005, 0.420, (10, 11)), (1.5, 0.004, 0.420, (10, 11)),
         (3.0, 0.0025, 0.465, (16, 17)), (3.0, 0.002, 0.465, (16, 17, 18))]
TAB = {0.008: tabs}
for (L, eta, th, rs) in CASES:
    if eta not in TAB: TAB[eta] = all_tables(eta, n)
    for r in rs:
        say("  L=%.1f r=%d eta=%.4f th=%.3f" % (L, r, eta, th))
        for m in range(5):
            say("     m=%d  %s" % (m, fmt(margin(r, L, eta, th, TAB[eta][m][0], TAB[eta][m][1], n))))
    say("  " + clock())

# ------------------------------------------------------------------ (iv) retuning (eta, theta) for the failing r
say("\n(iv) retuning eta, theta for m = 2, 3, 4 (n = 400); best margin per (L, r, m)")
SCAN = [(1.0, (7, 8), [0.006, 0.008, 0.010, 0.012], np.arange(0.350, 0.3951, 0.005)),
        (1.5, (10, 11), [0.003, 0.004, 0.005, 0.006], np.arange(0.400, 0.4451, 0.005)),
        (3.0, (16, 17), [0.0015, 0.002, 0.0025, 0.003], np.arange(0.445, 0.4851, 0.005))]
BEST = {}
for (L, rs, etas, ths) in SCAN:
    for eta in etas:
        if eta not in TAB: TAB[eta] = all_tables(eta, n)
        for th in ths:
            th = round(th, 4)
            for r in rs:
                for m in range(1, 5):
                    b = margin(r, L, eta, th, TAB[eta][m][0], TAB[eta][m][1], n)
                    key = (L, r, m)
                    if key not in BEST or b["M"] > BEST[key]["M"]:
                        BEST[key] = dict(b, eta=eta, th=th)
        say("  L=%.1f eta=%.4f done %s" % (L, eta, clock()))
    for r in rs:
        for m in range(1, 5):
            b = BEST[(L, r, m)]
            say("  L=%.1f r=%d m=%d  eta %.4f th %.3f  %s" % (L, r, m, b["eta"], b["th"], fmt(b)))
TAB.clear()

# ------------------------------------------------------------------ (v) grid refinement at the retuned optima of m = 2
say("\n(v) grid refinement n = 200, 400, 800 (h = 0.0025, 0.00125, 0.000625) at the m = 2 optima of the failing r")
for (L, r) in [(1.0, 7), (1.0, 8), (1.5, 10), (1.5, 11), (3.0, 16), (3.0, 17)]:
    b = BEST[(L, r, 2)]
    eta, th = b["eta"], b["th"]
    say("  L=%.1f r=%d eta=%.4f th=%.3f" % (L, r, eta, th))
    for nn in (200, 400, 800):
        if abs(round(th / (0.5 / nn)) * (0.5 / nn) - th) > 1e-9: continue
        tabs = all_tables(eta, nn)
        say("     n=%d: " % nn + "  ".join("m=%d M %+.4f" % (m, margin(r, L, eta, th, tabs[m][0], tabs[m][1], nn)["M"]) for m in range(5)) + "  " + clock())
# ------------------------------------------------------------------ (vi) towards the fixed point: m up to 12
say("\n(vi) m = 1..12 at the retuned optima (n = 400, and n = 800 for L = 3): margins M_m; eta -> 0 error at m = 12")
MM = 12
tabs = all_tables(1e-7, 400, MM)
n = 400; h = 0.5 / n
say("  eta = 1e-7, n = 400: Phi_12 - f at (1/2, 0.37) %+.5f, (1/2, 0.42) %+.5f, (1/2, 0.465) %+.5f; max over the table %+.5f; min(G_12 - F) over the region %+.5f" % (
    tabs[MM][0][n, int(round(0.37 / h))] - ss.ff(0.5 / 0.37)[0], tabs[MM][0][n, int(round(0.42 / h))] - ss.ff(0.5 / 0.42)[0],
    tabs[MM][0][n, int(round(0.465 / h))] - ss.ff(0.5 / 0.465)[0], (tabs[MM][0] - tabs[1][0]).max(),
    (tabs[MM][1][1:, 1:n][reg] - FE).min()))
for (L, rs, eta, th, nn) in [(1.0, (6, 7), 0.012, 0.365, 400), (1.0, (6, 7), 0.012, 0.365, 800), (1.5, (9, 10), 0.006, 0.420, 400),
                            (1.5, (9, 10), 0.006, 0.420, 800), (3.0, (15, 16), 0.003, 0.465, 400), (3.0, (15, 16), 0.003, 0.465, 800)]:
    tabs = all_tables(eta, nn, MM)
    for r in rs:
        say("  L=%.1f r=%d eta=%.4f th=%.3f n=%d: " % (L, r, eta, th, nn) + " ".join(
            "%+.4f" % margin(r, L, eta, th, tabs[m][0], tabs[m][1], nn)["M"] for m in range(1, MM + 1)) + "  " + clock())

# ------------------------------------------------------------------ (vii) retuning for r = 6, 9, 15 at m = 8
say("\n(vii) retuning eta, theta at m = 8 (n = 400) for the next r down")
for (L, rs, etas, ths) in [(1.0, (6, 7), [0.010, 0.012, 0.015, 0.018], np.arange(0.345, 0.3951, 0.005)),
                           (1.5, (9, 10), [0.005, 0.006, 0.008, 0.010], np.arange(0.395, 0.4451, 0.005)),
                           (3.0, (15, 16), [0.0025, 0.003, 0.004, 0.005], np.arange(0.445, 0.4851, 0.005))]:
    B8 = {}
    for eta in etas:
        tabs = all_tables(eta, 400, 8)
        for th in ths:
            th = round(th, 4)
            for r in rs:
                for m in (4, 8):
                    b = margin(r, L, eta, th, tabs[m][0], tabs[m][1], 400)
                    if (r, m) not in B8 or b["M"] > B8[(r, m)]["M"]: B8[(r, m)] = dict(b, eta=eta, th=th)
    for r in rs:
        for m in (4, 8):
            b = B8[(r, m)]
            say("  L=%.1f r=%d m=%d  eta %.4f th %.3f  %s" % (L, r, m, b["eta"], b["th"], fmt(b)))
    say("  " + clock())
say("\ndone " + clock())
