"""The fixed point of the Buchstab recursion (Lemma 4.3 of Part XIII) in one causal sweep, O(n^2) per table.

Row i (level X = i h) of Phi_m and G_m only reads rows i - l <= i - 1 of G_{m-1} and Phi_{m-1}. Sweeping i upwards
with the rows below already at their fixed point therefore gives the m = infinity fixed point in one pass (the
iteration of iterated_buchstab.py reaches it exactly at m = n).  Within a row, with u(i,j) = Phi(i,j)/sqrt(a_j),
  u(i,j) = max_{k <= j} [ c_k - (K_j - K_k) ] = cummax(c + K)_j - K_j,   c_k = Phi_0(i,k)/sqrt(a_k),
  K_j = sum_{l < j} (a_l^{-1/2} - a_{l+1}^{-1/2}) cell(i,l),  cell = mean (trapezoid) or max (one-sided) of the two
  endpoint values G(X_i - a_l; a_l), G(X_i - a_{l+1}; a_{l+1});  likewise G with cummin and the min of the Phi endpoints.
Phi(i,j) = 0 for j >= i (no room).  Phi_0 and G_0 are maxima/minima over s_1 grids, hence rigorous on any grid; here
the grids have nS points (coarser than the 400/600 of iterated_buchstab.py, which only weakens the bounds slightly).

Modes:  python causal_fixed_point.py check     (n = 400 against the m-iteration, eta -> 0 at n = 1600)
        python causal_fixed_point.py run       (r = 9, 15 and the Table 1 rows at n = 1600, 3200, 6400, both rules)
Output: results/causal_fixed_point_<mode>.txt"""
import os, sys, time, importlib.util
import numpy as np
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
sys.argv = ["x"]
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ss", os.path.join(HERE, "semilinear_sieve.py"))
ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
out = open(os.path.join(HERE, "..", "results", "causal_fixed_point_%s.txt" % MODE), "w", encoding="utf8")
def say(s=""):
    print(s, flush=True); out.write(s + "\n"); out.flush()
f12, F12, lk, c = ss.f12, ss.F12, ss.lk, ss.c
T0 = time.time()
def clock(): return "[%6.0f s]" % (time.time() - T0)

def rect_row(i, eta, n, nS, u):
    """Phi_0(i, j), G_0(i, j) for j = 1..n-1 at X = i h"""
    h = 0.5 / n; X = i * h; a = np.arange(1, n) * h
    P = np.zeros(n - 1)
    smax = (X - a) / eta
    ok = np.nonzero(smax > 1.0)[0]
    if len(ok):
        s1 = smax[ok, None] ** u[None, :]
        s3 = (X - eta * s1) / a[ok, None]
        f1, F1, f3, F3 = lk(f12, s1), lk(F12, s1), lk(f12, s3), lk(F12, s3)
        P[ok] = np.maximum(0.0, (f1 * F3 + F1 * f3 - F1 * F3).max(axis=1))
    s1 = np.geomspace(1e-4, X / eta * (1 - 1e-6), nS)
    F1 = lk(F12, s1)
    s3 = (X - eta * s1[None, :]) / a[:, None]
    F3 = np.where(s3 >= 1e-4, lk(F12, s3), np.inf)
    G = (F1[None, :] * F3).min(axis=1)
    return P, G

def causal(eta, n, safe, nS=120, rect=None):
    """fixed-point tables Phi[i, j], G[i, j] (index 0 unused); rect = optional precomputed (P0, G0) tables"""
    h = 0.5 / n
    a = np.arange(0, n + 1) * h
    ra = np.zeros(n + 1); ra[1:] = a[1:] ** -0.5; sa = np.sqrt(a)
    wc = ra[1:n - 1] - ra[2:n]                     # cells l = 1..n-2: a_l^{-1/2} - a_{l+1}^{-1/2}
    u = np.linspace(0.0, 1.0, nS)
    Phi = np.zeros((n + 1, n + 1)); G = np.zeros((n + 1, n + 1))
    for i in range(1, n + 1):
        if rect is None: P0, G0 = rect_row(i, eta, n, nS, u)
        else: P0, G0 = rect[0][i, 1:n], rect[1][i, 1:n]
        # endpoint values on the cells: H_l = G(X_i - a_l; a_l) = G[i - l, l], l = 1..i-1 (level > 0), else undefined
        l = np.arange(1, n)
        Hg = np.zeros(n - 1); Hp = np.zeros(n - 1)
        ll = l[l < i]
        Hg[:len(ll)] = G[i - ll, ll]; Hp[:len(ll)] = Phi[i - ll, ll]
        if safe:
            cg = np.maximum(Hg[:-1], Hg[1:]); cp = np.minimum(Hp[:-1], Hp[1:])
        else:
            cg = 0.5 * (Hg[:-1] + Hg[1:]); cp = 0.5 * (Hp[:-1] + Hp[1:])
        Kg = np.concatenate([[0.0], np.cumsum(wc * cg)])      # K_j, j = 1..n-1
        Kp = np.concatenate([[0.0], np.cumsum(wc * cp)])
        cvec = P0 * ra[1:n]; dvec = G0 * ra[1:n]
        uu = np.maximum.accumulate(cvec + Kg) - Kg
        vv = np.minimum.accumulate(dvec + Kp) - Kp
        row = sa[1:n] * uu
        row[np.arange(1, n) >= i] = 0.0                       # a_j >= X_i: no lower bound beyond 0 (and cells with G at level <= 0 are not used)
        Phi[i, 1:n] = row
        G[i, 1:n] = sa[1:n] * vv
    return Phi, G

JP = {}
def jparts(eta, th):
    hw = 5e-4 if eta < 0.006 else 1e-3
    key = (eta, round(th, 5), hw)
    if key not in JP: JP[key] = ss.Jparts(eta, th, hw)
    return JP[key]

def margin(r, L, eta, th, P, G, n, nT=240, nu=300):
    h = 0.5 / n; j = int(round(th / h))
    assert abs(j * h - th) < 1e-9
    Xg = np.arange(1, n + 1) * h
    ts = np.geomspace(eta, 0.49, nT)
    Gt = np.interp(0.5 - ts, Xg, G[1:, j]); Ph = P[n, j]
    parts = jparts(eta, th); j0 = ss.Jlam(parts, 0.0, 1.0); best = None
    for u0 in np.linspace(eta + 0.01, 0.48, nu):
        il = r + 1 - 1e-6 - 1 / u0
        if il <= 0: continue
        lam = 1 / il; k = ts <= u0
        if k.sum() < 2: continue
        integ = np.trapz((1 - ts[k] / u0) * Gt[k] / ts[k], ts[k])
        main = c * (Ph - lam / 2 * integ) / np.sqrt(eta * th)
        jj = ss.Jlam(parts, lam, u0) if 1 - 2 * th <= u0 else j0
        M = main - 8 * L * jj
        if best is None or M > best["M"]: best = dict(M=M, main=main, T=8 * L * jj, u0=u0, ilam=1 / lam, Phi=Ph)
    return best
def fmt(b): return "M %+.4f (%+5.1f%%)  main %.4f  sw %.4f  Phi %.5f  u0 %.3f  1/lam %.2f" % (b["M"], 100 * b["M"] / b["main"], b["main"], b["T"], b["Phi"], b["u0"], b["ilam"])

if MODE == "check":
    say("causal fixed point: checks " + clock())
    # (a) against the m-iteration of iterated_buchstab.py at n = 400 (same rect grids nS = 400/600 there; here nS = 120 and also 400)
    spec2 = importlib.util.spec_from_file_location("ib", os.path.join(HERE, "iterated_buchstab.py"))
    src = open(os.path.join(HERE, "iterated_buchstab.py"), encoding="utf8").read().split("# ------------------------------------------------------------------ (viii)")[0]
    src = src.replace('"iterated_buchstab%s.txt" % ("_" + MODE if MODE != "all" else "")', '"tmp_check.txt"')
    ib = {"__file__": os.path.join(HERE, "iterated_buchstab.py")}; exec(src, ib)
    n = 400; eta, th, r, L = 0.011, 0.4175, 9, 1.5
    for safe in (False, True):
        tabs = ib["all_tables"](eta, n, 40, safe)            # m up to 40
        rect = (tabs[0][0], tabs[0][1])
        Pc, Gc = causal(eta, n, safe, rect=rect)            # same rectangle tables -> must agree with m -> infinity
        for m in (12, 20, 30, 40):
            say("  n=400 %s m=%2d: max|Phi_m - Phi_causal| %.2e  max|G_m - G_causal| %.2e  M_m %+.5f" % (
                "safe" if safe else "trap", m, np.abs(tabs[m][0][1:, 1:n] - Pc[1:, 1:n]).max(), np.abs(tabs[m][1][1:, 1:n] - Gc[1:, 1:n]).max(),
                ib["margin"](r, L, eta, th, tabs[m][0], tabs[m][1], n)["M"]))
        say("  causal (rect 400/600): M %+.5f;  causal with its own rect grids nS=120: M %+.5f  %s" % (
            margin(r, L, eta, th, Pc, Gc, n)["M"], margin(r, L, eta, th, *causal(eta, n, safe, 120), n)["M"], clock()))
    ib["out"].close(); os.remove(os.path.join(HERE, "..", "results", "tmp_check.txt"))
    # (b) eta -> 0 at n = 1600, both rules
    for safe in (False, True):
        n = 1600; h = 0.5 / n
        P, G = causal(1e-7, n, safe, 120)
        say("  eta=1e-7 n=1600 %s: Phi - f at (1/2,.37) %+.5f (1/2,.42) %+.5f (1/2,.46) %+.5f; G - F at (.15,.35) %+.5f  %s" % (
            "safe" if safe else "trap", P[n, int(round(.37 / h))] - ss.ff(.5 / .37)[0], P[n, int(round(.42 / h))] - ss.ff(.5 / .42)[0],
            P[n, int(round(.46 / h))] - ss.ff(.5 / .46)[0], G[int(round(.15 / h)), int(round(.35 / h))] - ss.ff(.15 / .35)[1], clock()))
    say("done " + clock()); sys.exit(0)

if MODE == "boundary":
    # the optimal cuts (free boundary) of the fixed point at n = 1600, trapezoid rule, eta = 0.011 and 0.005
    say("optimal cuts of the fixed point, n = 1600 " + clock())
    for eta in (0.011, 0.005):
        n = 1600; h = 0.5 / n; nS = 120
        a = np.arange(0, n + 1) * h; ra = np.zeros(n + 1); ra[1:] = a[1:] ** -0.5
        wc = ra[1:n - 1] - ra[2:n]; u = np.linspace(0.0, 1.0, nS)
        Phi, G = causal(eta, n, False, nS)
        say("  eta = %.3f" % eta)
        for X in (0.5, 0.4, 0.3, 0.2, 0.15):
            i = int(round(X / h)); P0, G0 = rect_row(i, eta, n, nS, u)
            l = np.arange(1, n); Hg = np.zeros(n - 1); Hp = np.zeros(n - 1); ll = l[l < i]
            Hg[:len(ll)] = G[i - ll, ll]; Hp[:len(ll)] = Phi[i - ll, ll]
            Kg = np.concatenate([[0.0], np.cumsum(wc * 0.5 * (Hg[:-1] + Hg[1:]))]); Kp = np.concatenate([[0.0], np.cumsum(wc * 0.5 * (Hp[:-1] + Hp[1:]))])
            cg = P0 * ra[1:n] + Kg; cp = G0 * ra[1:n] + Kp
            kP = np.array([int(np.argmax(cg[:j])) + 1 for j in range(1, n)])      # cut index for Phi(X; a_j)
            kG = np.array([int(np.argmin(cp[:j])) + 1 for j in range(1, n)])
            cols = [0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.37, 0.42, 0.46]
            say("    X=%.2f  Phi cut theta0(a): " % X + "  ".join("a=%.2f->%s" % (aj, "%.4f" % (kP[int(round(aj / h)) - 1] * h) if aj < X else "--") for aj in cols))
            say("           G   cut theta0(a): " + "  ".join("a=%.2f->%.4f" % (aj, kG[int(round(aj / h)) - 1] * h) for aj in cols))
            # free boundary: largest a below which the Phi cut is trivial (theta0 = a) / nontrivial
            nz = [j for j in range(1, min(i, n)) if kP[j - 1] < j]
            say("           Phi: cut < a for a in [%.4f, %.4f] (%d of %d columns)" % (nz[0] * h, nz[-1] * h, len(nz), min(i, n) - 1) if nz else "           Phi: cut always trivial")
    say("done " + clock()); sys.exit(0)

if MODE == "lowmem":
    # Memory-light sweep for large n.  Rows of Phi_0, G_0 are generated on the fly.  Half-level scaling: for a >= X/2 every
    # s3 = (X - e s1)/a is < 2, where F(s3) = 2e^{gamma/2}/sqrt(pi s3) is proportional to sqrt(a); so G_0(X; e, a)/sqrt(a) is
    # constant for a >= X/2, and since Phi(X - b; e, b) = 0 for b >= X/2 the cells of the G-recursion vanish there: the fixed
    # point satisfies G(X; e, a) = sqrt(a / a_half) G(X; e, a_half) for a >= a_half = X/2 (exactly, on the grid as well).
    # Storage by anti-diagonals s = i + j <= n (row i' reads exactly the anti-diagonal s = i'): G is stored for l <= (s+1)/3
    # and otherwise given by the scalar v_{i} = G(i, ceil(i/2))/sqrt(a_{ceil(i/2)}); Phi is stored for l < s/2 (zero beyond).
    # Env: CFP_LM="L,r,eta,theta", CFP_N="1600,12800", CFP_RULES="safe,trap".  Output: results/causal_fixed_point_lowmem.txt
    L, r, eta, th = [float(x) for x in os.environ.get("CFP_LM", "3.0,15,0.005,0.4625").split(",")]; r = int(r)
    NS = [int(x) for x in os.environ.get("CFP_N", "1600,12800").split(",")]
    RULES = os.environ.get("CFP_RULES", "safe,trap").split(",")
    say("low-memory causal sweep with the half-level scaling: L=%.1f r=%d eta=%.4f th=%.4f " % (L, r, eta, th) + clock())
    def sweep_lowmem(n, safe, nS=120):
        h = 0.5 / n; jt = int(round(th / h)); assert abs(jt * h - th) < 1e-9
        a = np.arange(0, n + 2) * h; ra = np.zeros(n + 2); ra[1:] = a[1:] ** -0.5; sa = np.sqrt(a)
        u = np.linspace(0.0, 1.0, nS)
        s = np.arange(0, n + 2)
        LG = np.clip((s + 1) // 3, 0, None); LP = np.clip((s - 1) // 2, 0, None)      # stored lengths of diagonal s
        offG = np.concatenate([[0], np.cumsum(LG)]).astype(np.int64); offP = np.concatenate([[0], np.cumsum(LP)]).astype(np.int64)
        Gf = np.zeros(offG[n + 1], np.float32); Pf = np.zeros(offP[n + 1], np.float32)
        v = np.zeros(n + 1)                           # v[i] = G(i, ceil(i/2)) / sqrt(a_{ceil(i/2)})
        Phn = 0.0
        for i in range(1, n + 1):
            X = i * h; J = max(min(i - 1, jt), 1); jg = (i + 1) // 2
            # rectangle rows: Phi_0 for j = 1..J, G_0 for j = 1..jg then scaled
            aj = a[1:J + 1]; P0 = np.zeros(J)
            smax = (X - aj) / eta; ok = np.nonzero(smax > 1.0)[0]
            if len(ok):
                s1 = smax[ok, None] ** u[None, :]; s3 = (X - eta * s1) / aj[ok, None]
                P0[ok] = np.maximum(0.0, (lk(f12, s1) * lk(F12, s3) + lk(F12, s1) * lk(f12, s3) - lk(F12, s1) * lk(F12, s3)).max(axis=1))
            s1 = np.geomspace(1e-4, X / eta * (1 - 1e-6), nS); F1 = lk(F12, s1)
            s3 = (X - eta * s1[None, :]) / a[1:jg + 1, None]
            G0g = (F1[None, :] * np.where(s3 >= 1e-4, lk(F12, s3), np.inf)).min(axis=1)
            # cells l = 1..J-1: endpoint values G(i-l, l), Phi(i-l, l) from the anti-diagonal s = i
            l = np.arange(1, J + 1)
            Hg = np.zeros(J); Hp = np.zeros(J)
            if i >= 2:
                m = min(J, i - 1); ll = l[:m]
                nG = min(m, LG[i]); Hg[:nG] = Gf[offG[i]:offG[i] + nG]
                if m > nG: Hg[nG:m] = sa[ll[nG:]] * v[i - ll[nG:]]
                nP = min(m, LP[i]); Hp[:nP] = Pf[offP[i]:offP[i] + nP]
            wc = ra[1:J] - ra[2:J + 1]
            if safe: cg = np.maximum(Hg[:-1], Hg[1:]); cp = np.minimum(Hp[:-1], Hp[1:])
            else: cg = 0.5 * (Hg[:-1] + Hg[1:]); cp = 0.5 * (Hp[:-1] + Hp[1:])
            Kg = np.concatenate([[0.0], np.cumsum(wc * cg)]); Kp = np.concatenate([[0.0], np.cumsum(wc * cp)])
            rowP = sa[1:J + 1] * (np.maximum.accumulate(P0 * ra[1:J + 1] + Kg) - Kg); rowP[l >= i] = 0.0
            y = G0g * ra[1:jg + 1] + Kp[:jg]
            rowG = sa[1:jg + 1] * (np.minimum.accumulate(y) - Kp[:jg])
            v[i] = rowG[jg - 1] * ra[jg]
            if i == n: Phn = rowP[jt - 1] if jt <= J else 0.0
            # store diagonal pieces: (i, j) -> diagonal i + j, position j - 1
            jmG = min(jg, LG[i + 1:i + jg + 1].max() if jg else 0, n - i)
            jj = np.arange(1, min(jg, n - i) + 1); keep = jj <= LG[i + jj]
            if keep.any(): Gf[offG[i + jj[keep]] + jj[keep] - 1] = rowG[jj[keep] - 1]
            jj = np.arange(1, min(J, n - i) + 1); keep = (jj <= LP[i + jj]) & (jj < i)
            if keep.any(): Pf[offP[i + jj[keep]] + jj[keep] - 1] = rowP[jj[keep] - 1]
            if i % 2000 == 0: say("    row %d of %d %s" % (i, n, clock()))
        Gcol = sa[jt] * v                                   # G(i, j_theta) = sqrt(a_{j_theta}) v_i  (j_theta >= ceil(i/2) for all i <= n)
        return Gcol, Phn, Gf.nbytes + Pf.nbytes
    def margin_col(Gcol, Ph, n, nT=240, nu=300):
        h = 0.5 / n; Xg = np.arange(1, n + 1) * h
        ts = np.geomspace(eta, 0.49, nT); Gt = np.interp(0.5 - ts, Xg, Gcol[1:n + 1])
        parts = jparts(eta, th); j0 = ss.Jlam(parts, 0.0, 1.0); best = None
        for u0 in np.linspace(eta + 0.01, 0.48, nu):
            il = r + 1 - 1e-6 - 1 / u0
            if il <= 0: continue
            lam = 1 / il; k = ts <= u0
            if k.sum() < 2: continue
            integ = np.trapz((1 - ts[k] / u0) * Gt[k] / ts[k], ts[k])
            main = c * (Ph - lam / 2 * integ) / np.sqrt(eta * th)
            jj = ss.Jlam(parts, lam, u0) if 1 - 2 * th <= u0 else j0
            M = main - 8 * L * jj
            if best is None or M > best["M"]: best = dict(M=M, main=main, T=8 * L * jj, u0=u0, ilam=1 / lam, Phi=Ph)
        return best
    for n in NS:
        for rule in RULES:
            Gcol, Ph, nb = sweep_lowmem(n, rule == "safe")
            b = margin_col(Gcol, Ph, n)
            say("  n=%d %s: %s   (tables %.0f MB)  %s" % (n, rule, fmt(b), nb / 1e6, clock()))
    say("done " + clock()); sys.exit(0)

if MODE == "run":
    say("causal fixed point on n = 1600, 3200, 6400, both rules (nS = 120) " + clock())
    CASES = [(1.5, 9, 0.011, 0.4175), (3.0, 15, 0.005, 0.4625), (1.0, 7, 0.015, 0.370), (1.5, 10, 0.010, 0.420), (3.0, 16, 0.005, 0.460), (1.0, 6, 0.015, 0.370)]
    if os.environ.get("CFP_CASES"):          # e.g. CFP_CASES="3.0,15,0.005,0.4625;1.5,9,0.011,0.4175"
        CASES = [tuple(float(v) if k != 1 else int(v) for k, v in enumerate(c.split(","))) for c in os.environ["CFP_CASES"].split(";")]
    NS = [int(x) for x in os.environ.get("CFP_N", "1600,3200,6400").split(",")]
    RECT = {}
    def rect_tables(eta, n, nS=120):
        """full Phi_0, G_0 tables, computed once per (eta, n) and shared by both rules; None if memory is short"""
        if (eta, n) in RECT: return RECT[(eta, n)]
        RECT.clear()
        if n > 6400: return None                 # 2 x 1.3 GB would be needed; recompute the rows per rule instead
        try:
            P0 = np.zeros((n + 1, n + 1)); G0 = np.zeros((n + 1, n + 1))
        except MemoryError:
            return None
        u = np.linspace(0.0, 1.0, nS)
        for i in range(1, n + 1):
            P0[i, 1:n], G0[i, 1:n] = rect_row(i, eta, n, nS, u)
        RECT[(eta, n)] = (P0, G0); return RECT[(eta, n)]
    for (L, r, eta, th) in CASES:
        vals = {}
        for n in NS:
            if abs(round(th / (0.5 / n)) * (0.5 / n) - th) > 1e-9: continue
            rect = rect_tables(eta, n)
            say("  rectangle tables eta=%.4f n=%d ready %s" % (eta, n, clock()))
            for safe in (False, True):
                P, G = causal(eta, n, safe, 120, rect)
                b = margin(r, L, eta, th, P, G, n); vals[(n, safe)] = b["M"]
                say("  L=%.1f r=%2d eta=%.4f th=%.4f n=%5d %s: %s  %s" % (L, r, eta, th, n, "safe" if safe else "trap", fmt(b), clock()))
                del P, G
        for safe in (False, True):
            v = [vals.get((n, safe)) for n in NS]
            if None in v or len(v) < 3: continue
            d1, d2 = v[1] - v[0], v[2] - v[1]
            if d1 * d2 > 0 and abs(d2) < abs(d1):
                p = np.log2(d1 / d2)
                say("    Richardson (%s): order %.2f, limit %+.4f (p = 1: %+.4f)" % ("safe" if safe else "trap", p, v[2] + d2 / (2 ** p - 1), v[2] + d2))
            else:
                say("    Richardson (%s): differences %+.4f %+.4f not geometric" % ("safe" if safe else "trap", d1, d2))
    say("done " + clock()); sys.exit(0)
