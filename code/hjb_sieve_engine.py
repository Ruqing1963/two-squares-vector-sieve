"""Coupled two-set sieve engine: the fixed point of the Buchstab recursion on the main set in one causal sweep.

Setting.  An auxiliary sifting set of dimension kappa_aux in {1/2, 1} up to N^e (sieve functions f_aux, F_aux) and a
main sifting set of dimension 1/2 up to N^a (half-dimensional f, F).  Rectangle (vector-sieve) bounds at level N^X:
  Phi_0(X; e, a) = max_{e s1 + a s3 = X, s1, s3 >= 1} [f_aux(s1) F(s3) + F_aux(s1) f(s3) - F_aux(s1) F(s3)]^+,
  G_0(X; e, a)   = min_{e s1 + a s3 = X} F_aux(s1) F(s3).
Buchstab steps on the main set (cut theta0 <= a, inert density 1/2 and V(N^a) ~ a^{-1/2}) give the recursion of
Part XIII, Lemma 4.3; its fixed point (Phi, G) satisfies, with u = a^{-1/2} Phi and v = a^{-1/2} G,
  u(X, a) = max_{theta0 <= a} [ psi^-(X, theta0) - 1/2 int_{theta0}^{a} b^{-3/2} G(X - b; e, b) db ],   psi^- = a^{-1/2} Phi_0,
  v(X, a) = min_{theta0 <= a} [ psi^+(X, theta0) - 1/2 int_{theta0}^{a} b^{-3/2} Phi(X - b; e, b) db ],  psi^+ = a^{-1/2} G_0,
i.e. an obstacle problem: either u = psi^- (stop) or -a d_a u = (1/2) a^{-1/2} G(X - a; e, a) (continue), and likewise for v.
Causality: row X = i h only reads rows X - b < X, so one upward sweep in i gives the exact fixed point (the m-step
iteration reaches it at m = n).  Within a row the cut search is a cumulative max/min.  Grid X = ih, a = jh, h = 1/(2n);
cell measure b_l^{-1/2} - b_{l+1}^{-1/2} times the mean (trapezoid) or the safe extreme (one-sided) of the endpoint values.
Phi_0, G_0 are maxima/minima over s1-grids, hence valid bounds on every grid (nS points).

Modes:
  verify   : n = 400 and 1600, engine against the 12-step iteration of iterated_buchstab.py (error and time)
  phase    : optimal-cut maps (free boundary) at n = 1600 for e = 0.011 and 0.005; figures/hjb_phase.pdf
  app2     : margins with the switched term multiplied by the Lichtman-Li factor (results/lichtman_factor.npy or argv[2])
Output: results/hjb_sieve_engine_<mode>.txt"""
import os, sys, time, importlib.util
import numpy as np
MODE = sys.argv[1] if len(sys.argv) > 1 else "verify"
ARG2 = sys.argv[2] if len(sys.argv) > 2 else None
sys.argv = ["x"]
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); FIG = os.path.join(HERE, "..", "figures")
spec = importlib.util.spec_from_file_location("ss", os.path.join(HERE, "semilinear_sieve.py"))
ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
out = open(os.path.join(RES, "hjb_sieve_engine_%s.txt" % MODE), "w", encoding="utf8") if __name__ == "__main__" else None
def say(s=""):
    print(s, flush=True)
    if out: out.write(s + "\n"); out.flush()
T0 = time.time()
def clock(): return "[%6.0f s]" % (time.time() - T0)
c = ss.c                                   # e^{-gamma}

# ---------------- sieve functions: half-dimensional (from semilinear_sieve) and linear ----------------
hS = ss.h; SGRID = ss.S; NS12 = len(SGRID)
def _linear():
    g = np.euler_gamma; C = 2 * np.exp(g); S = SGRID; n = len(S); lag = int(round(1 / hS))
    F = np.where(S <= 3 + 1e-12, C / S, 0.0)
    f = np.where((S >= 2 - 1e-12) & (S <= 4 + 1e-12), C * np.log(np.maximum(S - 1, 1)) / S, 0.0)
    sF = S * F; sf = S * f
    i3 = int(round(3 / hS)) - 1; i4 = int(round(4 / hS)) - 1
    for i in range(i3 + 1, n):
        sF[i] = sF[i - 1] + hS * 0.5 * (f[i - lag] + f[i - 1 - lag]); F[i] = sF[i] / S[i]
        if i > i4:
            sf[i] = sf[i - 1] + hS * 0.5 * (F[i - lag] + F[i - 1 - lag]); f[i] = sf[i] / S[i]
    return f, F
F1TAB = None
def sieve_tables(kappa):
    global F1TAB
    if kappa == 0.5: return ss.f12, ss.F12
    if F1TAB is None: F1TAB = _linear()
    return F1TAB
lk = ss.lk

# ---------------- the engine ----------------
class Engine:
    def __init__(self, e, n, kappa_aux=0.5, nS=120):
        self.e, self.n, self.kappa_aux, self.nS = e, n, kappa_aux, nS
        self.h = 0.5 / n
        self.fa, self.Fa = sieve_tables(kappa_aux)
        self.f, self.F = sieve_tables(0.5)
        self.a = np.arange(0, n + 1) * self.h
        self.ra = np.zeros(n + 1); self.ra[1:] = self.a[1:] ** -0.5
        self.sa = np.sqrt(self.a)
        self.wc = self.ra[1:n - 1] - self.ra[2:n]
        self.ugrid = np.linspace(0.0, 1.0, nS)

    def rect_row(self, i):
        """Phi_0(i, j), G_0(i, j), j = 1..n-1"""
        n, e, h = self.n, self.e, self.h
        X = i * h; a = self.a[1:n]
        P = np.zeros(n - 1)
        smax = (X - a) / e
        ok = np.nonzero(smax > 1.0)[0]
        if len(ok):
            s1 = smax[ok, None] ** self.ugrid[None, :]
            s3 = (X - e * s1) / a[ok, None]
            f1, F1, f3, F3 = lk(self.fa, s1), lk(self.Fa, s1), lk(self.f, s3), lk(self.F, s3)
            P[ok] = np.maximum(0.0, (f1 * F3 + F1 * f3 - F1 * F3).max(axis=1))
        s1min = 1.0 if self.kappa_aux == 1.0 else 1e-4        # linear sieve: F valid for s >= beta - 1 = 1
        if X / e * (1 - 1e-6) <= s1min: return P, np.full(n - 1, np.inf)
        s1 = np.geomspace(s1min, X / e * (1 - 1e-6), self.nS)
        F1 = lk(self.Fa, s1)
        s3 = (X - e * s1[None, :]) / a[:, None]
        F3 = np.where(s3 >= 1e-4, lk(self.F, s3), np.inf)
        G = (F1[None, :] * F3).min(axis=1)
        return P, G

    def solve(self, rule="trapezoid", cuts=False, rect=None):
        """fixed-point tables Phi[i, j], G[i, j]; with cuts=True also the optimal cut indices kP, kG (int32)"""
        n, h = self.n, self.h
        Phi = np.zeros((n + 1, n + 1)); G = np.zeros((n + 1, n + 1))
        kP = np.zeros((n + 1, n + 1), np.int32) if cuts else None
        kG = np.zeros((n + 1, n + 1), np.int32) if cuts else None
        l = np.arange(1, n); idx = np.arange(1, n)
        for i in range(1, n + 1):
            if rect is None: P0, G0 = self.rect_row(i)
            else: P0, G0 = rect[0][i, 1:n], rect[1][i, 1:n]
            Hg = np.zeros(n - 1); Hp = np.zeros(n - 1)
            ll = l[l < i]
            Hg[:len(ll)] = G[i - ll, ll]; Hp[:len(ll)] = Phi[i - ll, ll]
            if rule == "one-sided":
                cg = np.maximum(Hg[:-1], Hg[1:]); cp = np.minimum(Hp[:-1], Hp[1:])
            else:
                cg = 0.5 * (Hg[:-1] + Hg[1:]); cp = 0.5 * (Hp[:-1] + Hp[1:])
            # cells whose G is infinite (no upper bound at that level, e.g. level below the auxiliary sieve when
            # kappa_aux = 1) cannot be crossed by a cut: beyond the first such cell the lower bound is Phi_0 itself
            bad = ~np.isfinite(cg)
            if bad.any():
                lbad = int(np.argmax(bad)) + 1; cg = np.where(bad, 0.0, cg)
            else:
                lbad = n
            Kg = np.concatenate([[0.0], np.cumsum(self.wc * cg)])
            Kp = np.concatenate([[0.0], np.cumsum(self.wc * cp)])
            x = P0 * self.ra[1:n] + Kg; y = G0 * self.ra[1:n] + Kp
            xm = np.maximum.accumulate(x); ym = np.minimum.accumulate(y)
            row = self.sa[1:n] * (xm - Kg); row[idx >= i] = 0.0
            if lbad < n: row[idx > lbad] = P0[idx > lbad]
            Phi[i, 1:n] = row
            G[i, 1:n] = self.sa[1:n] * (ym - Kp)
            if cuts:
                tmp = np.maximum.accumulate(np.where(x == xm, idx, 0)); tmp[idx >= i] = 0
                if lbad < n: tmp[idx > lbad] = idx[idx > lbad]
                kP[i, 1:n] = tmp
                kG[i, 1:n] = np.maximum.accumulate(np.where(y == ym, idx, 0))
        return (Phi, G, kP, kG) if cuts else (Phi, G)

JP = {}
def jparts(eta, th):
    hw = 5e-4 if eta < 0.006 else 1e-3
    key = (eta, round(th, 5), hw)
    if key not in JP: JP[key] = ss.Jparts(eta, th, hw)
    return JP[key]

def margin(r, L, eta, th, Phi, G, n, rho=0.5, sw_factor=1.0, nT=240, nu=300):
    """Part XIII criterion (kappa_aux = 1/2): M = e^{-gamma}/sqrt(eta th) [Phi - lam rho int (1 - t/u0) G dt/t] - sw_factor 8 L J_lam"""
    h = 0.5 / n; j = int(round(th / h))
    assert abs(j * h - th) < 1e-9, "theta must lie on the grid"
    Xg = np.arange(1, n + 1) * h
    ts = np.geomspace(eta, 0.49, nT)
    Gt = np.interp(0.5 - ts, Xg, G[1:, j]); Ph = Phi[n, j]
    parts = jparts(eta, th); j0 = ss.Jlam(parts, 0.0, 1.0); best = None
    for u0 in np.linspace(eta + 0.01, 0.48, nu):
        il = r + 1 - 1e-6 - 1 / u0
        if il <= 0: continue
        lam = 1 / il; k = ts <= u0
        if k.sum() < 2: continue
        integ = np.trapz((1 - ts[k] / u0) * Gt[k] / ts[k], ts[k])
        main = c * (Ph - lam * rho * integ) / np.sqrt(eta * th)
        T = sw_factor * 8 * L * (ss.Jlam(parts, lam, u0) if 1 - 2 * th <= u0 else j0)
        M = main - T
        if best is None or M > best["M"]: best = dict(M=M, main=main, T=T, u0=u0, ilam=1 / lam, Phi=Ph)
    return best
def fmt(b): return "M %+.4f (%+5.1f%%)  main %.4f  sw %.4f  Phi %.5f  u0 %.3f  1/lam %.2f" % (b["M"], 100 * b["M"] / b["main"], b["main"], b["T"], b["Phi"], b["u0"], b["ilam"])

# ====================================================================================================
if __name__ == "__main__" and MODE == "verify":
    say("engine against the 12-step iteration of iterated_buchstab.py " + clock())
    src = open(os.path.join(HERE, "iterated_buchstab.py"), encoding="utf8").read().split("# ------------------------------------------------------------------ (viii)")[0]
    src = src.replace('"iterated_buchstab%s.txt" % ("_" + MODE if MODE != "all" else "")', '"tmp_verify.txt"')
    ib = {"__file__": os.path.join(HERE, "iterated_buchstab.py")}; exec(src, ib)
    eta, th, r, L = 0.011, 0.4175, 9, 1.5
    for n in (400, 1600):
        for safe in (False, True):
            t = time.time(); tabs = ib["all_tables"](eta, n, 12, safe); told = time.time() - t
            rect = (tabs[0][0], tabs[0][1])
            eng = Engine(eta, n)
            t = time.time(); Pc, Gc = eng.solve("one-sided" if safe else "trapezoid", rect=rect); tnew = time.time() - t
            t = time.time(); P2, G2 = eng.solve("one-sided" if safe else "trapezoid"); tfull = time.time() - t
            dP = np.abs(tabs[12][0][1:, 1:n] - Pc[1:, 1:n]).max(); dG = np.abs(tabs[12][1][1:, 1:n] - Gc[1:, 1:n]).max()
            say("  n=%4d %s: 12-step iteration %.1f s (12 steps of the O(n^3) recursion, plus the rectangle tables); engine sweep %.2f s on the same rectangle tables (max |dPhi| %.1e, |dG| %.1e; the iteration is not yet at its fixed point at m = 12), %.1f s with its own nS=120 tables"
                % (n, "safe" if safe else "trap", told, tnew, dP, dG, tfull))
            say("        margins: iteration m=12 %+.5f, engine (same tables) %+.5f, engine (nS=120) %+.5f  %s" % (
                ib["margin"](r, L, eta, th, tabs[12][0], tabs[12][1], n)["M"], margin(r, L, eta, th, Pc, Gc, n)["M"], margin(r, L, eta, th, P2, G2, n)["M"], clock()))
            del tabs, rect, Pc, Gc, P2, G2
    ib["out"].close(); os.remove(os.path.join(RES, "tmp_verify.txt"))
    say("done " + clock())

if __name__ == "__main__" and MODE == "phase":
    say("free boundary of the fixed point, n = 1600, trapezoid rule " + clock())
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    n = 1600; h = 0.5 / n
    for col, e in enumerate((0.011, 0.005)):
        eng = Engine(e, n)
        Phi, G, kP, kG = eng.solve("trapezoid", cuts=True)
        # cut ratio theta0*/a for Phi where defined (j < i), else nan; stop region: kP == j
        J = np.arange(0, n + 1)[None, :]
        ratio = np.where((kP > 0), kP / np.maximum(J, 1), np.nan)
        stop = (kP == J) & (kP > 0)
        X = np.arange(0, n + 1) * h
        im = ax[col].imshow(np.log10(ratio[1:, 1:n]).T, origin="lower", extent=[h, 0.5, h, 0.5 - h], aspect="auto", cmap="viridis", vmin=-3.5, vmax=0)
        ax[col].contour(X[1:], X[1:n], stop[1:, 1:n].T.astype(float), levels=[0.5], colors="white", linewidths=1.2)
        ax[col].plot([0, 0.5], [0, 0.5], "k:", lw=0.8)
        ax[col].set_xlabel(r"level $X$"); ax[col].set_ylabel(r"inert exponent $a$"); ax[col].set_title(r"$\log_{10}(\theta_0^*/a)$ for $\Phi$, $e=%.3f$" % e)
        # boundary data: per X, the largest a in the active region, and the region where theta0* > h
        rows = []
        for X0 in (0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05):
            i = int(round(X0 / h)); act = [j for j in range(1, i) if kP[i, j] < j]
            deep = [j for j in range(1, i) if 1 < kP[i, j] < j]
            rows.append("X=%.2f: active a in [%.4f, %.4f] (X - a_max = %.4f = %.2f e); cut theta0* > h for %d columns%s" % (
                X0, act[0] * h, act[-1] * h, X0 - act[-1] * h, (X0 - act[-1] * h) / e, len(deep),
                "" if not deep else ", e.g. a=%.4f -> theta0*=%.4f" % (deep[-1] * h, kP[i, deep[-1]] * h)) if act else "X=%.2f: no active column" % X0)
        say("  e = %.3f:\n    " % e + "\n    ".join(rows))
        kGn = kG[n, 1:n]; say("    G at X = 1/2: cut index > 1 in %d of %d columns; max theta0* = %.4f" % ((kGn > 1).sum(), n - 1, kGn.max() * h))
        # the cut map for G at low levels
        for X0 in (0.1, 0.05, 0.03):
            i = int(round(X0 / h)); kk = kG[i, 1:n]
            say("    G at X = %.2f: theta0* > h in %d columns; at a = 0.3: theta0* = %.4f, at a = 0.1: %.4f" % (X0, (kk > 1).sum(), kG[i, int(round(0.3 / h))] * h, kG[i, int(round(0.1 / h))] * h))
    fig.colorbar(im, ax=ax, shrink=0.9, label=r"$\log_{10}(\theta_0^*/a)$")
    fig.savefig(os.path.join(FIG, "hjb_phase.pdf")); fig.savefig(os.path.join(FIG, "hjb_phase.png"), dpi=150)
    say("figure written " + clock())

if __name__ == "__main__" and MODE == "app2":
    fac = float(ARG2) if ARG2 else float(np.load(os.path.join(RES, "lichtman_factor.npy")))
    say("application 2: switched term times the Lichtman-Li factor %.4f (effective level %.4f) " % (fac, 1 / (2 * fac)) + clock())
    setups = {1.0: ((5, 6), [0.012, 0.015, 0.02, 0.025, 0.03], np.arange(0.33, 0.4051, 0.005)),
              1.5: ((8, 9), [0.008, 0.01, 0.012, 0.015, 0.02], np.arange(0.38, 0.4551, 0.005)),
              3.0: ((14, 15), [0.004, 0.005, 0.006, 0.008, 0.01], np.arange(0.43, 0.4951, 0.005))}
    n = 400
    for L, (rs, etas, ths) in setups.items():
        best = {}
        for eta in etas:
            Phi, G = Engine(eta, n).solve("trapezoid")
            for th in [round(x, 4) for x in ths]:
                for r in rs:
                    b = margin(r, L, eta, th, Phi, G, n, sw_factor=fac)
                    if r not in best or b["M"] > best[r]["M"]: best[r] = dict(b, eta=eta, th=th)
        for r in rs:
            b = best[r]; eta, th = b["eta"], b["th"]
            say("  L=%.1f r=%2d  n=400 trap: eta %.4f th %.4f  %s" % (L, r, eta, th, fmt(b)))
            for nn in (1600,):
                if abs(round(th / (0.5 / nn)) * (0.5 / nn) - th) > 1e-9: continue
                eng = Engine(eta, nn)
                for rule in ("trapezoid", "one-sided"):
                    Phi, G = eng.solve(rule)
                    say("        n=%d %s: %s  %s" % (nn, rule, fmt(margin(r, L, eta, th, Phi, G, nn, sw_factor=fac)), clock()))
    say("done " + clock())
