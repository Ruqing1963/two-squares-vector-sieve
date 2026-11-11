"""Part XIII: explicit r from the half-dimensional vector sieve with switching and Richert weights.

Quantities are in units of 2B, B = X * Sg(N) * Dcorr / L (Section 4 of the paper):
  main (unweighted)  = e^{-gamma} Phi(eta, th) / sqrt(eta th)
  main (weighted)    = e^{-gamma} [Phi - (lam/2) int_eta^u0 (1 - t/u0) G(t) dt/t] / sqrt(eta th)
  switched term      = 8 L J(eta, th)
Positivity  main - switched > 0  gives Omega(m) <= r with
  r = floor(1/eta) (unweighted, eta not 1/integer) or 1/lam + 1/u0 = r + 1 - 0 (weighted).
L = 1 (no class condition on the prime, or h = 1), L = 3/2 (h = 3, C' non-principal), L = 3 (h = 3, C' principal).

Run: python semilinear_sieve.py     (about 30 minutes; writes results/ and figures/)
"""
import os, sys, time
import numpy as np
from math import gamma as Gamma

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); FIG = os.path.join(HERE, "..", "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
out = open(os.path.join(RES, "semilinear_sieve.txt"), "w", encoding="utf8")
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n"); out.flush()

g = np.euler_gamma
EG = np.exp(g)
c = np.exp(-g)

# ---------------- semi-linear sieve functions (kappa = 1/2, beta = 1) ----------------
h = 1e-4
S = np.arange(h, 40.0 + h / 2, h)
lag = int(round(1.0 / h))
rt = np.sqrt(S)

def semilinear():
    n = len(S)
    F = np.where(S <= 2.0 + 1e-12, 2 * np.exp(g / 2) / np.sqrt(np.pi * S), 0.0)
    f = np.zeros(n)
    i1 = int(round(1.0 / h)) - 1; i2 = int(round(2.0 / h)) - 1; i3 = int(round(3.0 / h)) - 1
    sf = rt * f; sF = rt * F
    for i in range(i1 + 1, n):
        if i <= i3:
            s = S[i]
            f[i] = np.exp(g / 2) / np.sqrt(np.pi * s) * 2 * np.log(np.sqrt(s) + np.sqrt(s - 1))
            sf[i] = rt[i] * f[i]
        else:   # (s^{1/2} f)' = (1/2) s^{-1/2} F(s-1)
            sf[i] = sf[i - 1] + 0.5 * h * (0.5 / rt[i] * F[i - lag] + 0.5 / rt[i - 1] * F[i - 1 - lag])
            f[i] = sf[i] / rt[i]
        if i > i2:      # (s^{1/2} F)' = (1/2) s^{-1/2} f(s-1)
            sF[i] = sF[i - 1] + 0.5 * h * (0.5 / rt[i] * f[i - lag] + 0.5 / rt[i - 1] * f[i - 1 - lag])
            F[i] = sF[i] / rt[i]
    return f, F

f12, F12 = semilinear()
def ff(s):
    j = int(round(min(s, 39.9) / h)) - 1
    return f12[j], F12[j]

# ---------------- ingredients ----------------
def Phi(eta, th, step=0.005):
    best, arg = -1e9, None
    for s1 in np.arange(1.0, 39.0, step):
        s3 = (0.5 - eta * s1) / th
        if s3 < 1.0: break
        f1, F1 = ff(s1); f3, F3 = ff(s3)
        v = f1 * F3 + F1 * f3 - F1 * F3
        if v > best: best, arg = v, (s1, s3)
    return best, arg

def J(eta, th, hw=1e-3):
    a = np.arange(th, 0.5, hw) + hw / 2
    tot = np.sum(1.0 / (a * (1 - a))) * hw
    W = np.arange(hw / 2, 1 - 2 * th, hw)
    if len(W) == 0: return tot / 4
    Pk = np.where(W >= eta, 1.0 / W, 0.0); k = 1
    while True:
        inner = np.zeros_like(W)
        for i, w in enumerate(W):
            amax = (1 - w) / 2
            if amax <= th: continue
            aa = np.arange(th, amax, hw / 4) + hw / 8
            inner[i] = np.sum(1.0 / (aa * (1 - aa - w))) * hw / 4
        tot += 2.0 ** (-k) * np.sum(Pk * inner) * hw
        Pn = np.zeros_like(W)
        for i, w in enumerate(W):
            if w < (k + 1) * eta: continue
            t = W[(W >= eta) & (W <= w - k * eta + 1e-12)]
            idx = np.clip(np.searchsorted(W, w - t), 0, len(W) - 1)
            Pn[i] = np.sum(Pk[idx] / t) * hw / (k + 1)
        if not Pn.any(): break
        Pk = Pn; k += 1
    return tot / 4

def G(t, eta, th):
    rem = 0.5 - t; best = 1e9
    for s1 in np.linspace(0.02, rem / eta - 0.02, 150):
        s3 = (rem - eta * s1) / th
        if s3 <= 0.01: continue
        best = min(best, ff(s1)[1] * ff(s3)[1])
    return best

def Jparts(eta, th, hw):
    """J split by k: J_lam = 1/4 [t0 + sum_k sum_w terms_k(w) max(0, 1 - lam (k - w/u0))]."""
    a = np.arange(th, 0.5, hw) + hw / 2
    t0 = np.sum(1.0 / (a * (1 - a))) * hw
    W = np.arange(hw / 2, 1 - 2 * th, hw)
    inner = np.zeros_like(W)
    for i, w in enumerate(W):
        amax = (1 - w) / 2
        if amax <= th: continue
        aa = np.arange(th, amax, hw / 4) + hw / 8
        inner[i] = np.sum(1.0 / (aa * (1 - aa - w))) * hw / 4
    terms = []
    Pk = np.where(W >= eta, 1.0 / W, 0.0); k = 1
    while True:
        terms.append(2.0 ** (-k) * Pk * inner * hw)
        Pn = np.zeros_like(W)
        for i, w in enumerate(W):
            if w < (k + 1) * eta: continue
            t = W[(W >= eta) & (W <= w - k * eta + 1e-12)]
            idx = np.clip(np.searchsorted(W, w - t), 0, len(W) - 1)
            Pn[i] = np.sum(Pk[idx] / t) * hw / (k + 1)
        if not Pn.any(): break
        Pk = Pn; k += 1
    return t0, W, terms

def Jlam(parts, lam, u0):
    """switched-term integral with the Richert weight w_+(r) on the bad survivors (lam = 0: plain J)"""
    t0, W, terms = parts
    tot = t0
    for k, tk in enumerate(terms, start=1):
        tot += np.sum(tk * np.maximum(0.0, 1 - lam * (k - W / u0)))
    return tot / 4

cache = {}
def ingr(eta, th):
    key = (eta, round(th, 4))
    if key not in cache:
        P, arg = Phi(eta, th)
        parts = Jparts(eta, th, 5e-4 if eta < 0.006 else 1e-3)
        ts = np.geomspace(eta, 0.49, 240)
        Gs = np.array([G(t, eta, th) for t in ts]) if P > 0 else None
        cache[key] = (P, arg, parts, ts, Gs)
    return cache[key]

def weighted(r, L, etas, ths):
    """returns (best with weighted bad survivors, best with bad survivors at weight 1)"""
    best = None; old = None
    for eta in etas:
        for th in ths:
            P, arg, parts, ts, Gs = ingr(eta, th)
            if P <= 0: continue
            j0 = Jlam(parts, 0.0, 1.0)
            for u0 in np.linspace(eta + 0.01, 0.48, 300):
                il = r + 1 - 1e-6 - 1 / u0
                if il <= 0: continue
                lam = 1 / il; m = ts <= u0
                if m.sum() < 2: continue
                integ = np.trapz((1 - ts[m] / u0) * Gs[m] / ts[m], ts[m])
                main = c * (P - lam / 2 * integ) / np.sqrt(eta * th)
                jj = Jlam(parts, lam, u0) if 1 - 2 * th <= u0 else j0
                M = main - 8 * L * jj
                if best is None or M > best["M"]:
                    best = dict(M=M, eta=eta, th=th, u0=u0, lam=lam, s=arg, main=main, T=8 * L * jj)
                M0 = main - 8 * L * j0
                if old is None or M0 > old["M"]:
                    old = dict(M=M0, main=main, T=8 * L * j0)
    return best, old

if __name__ == "__main__":
    t0 = time.time()
    say("Part XIII semi-linear vector sieve numerics")
    a = ff(1.5); cl = np.exp(g / 2) / np.sqrt(np.pi * 1.5) * 2 * np.log(np.sqrt(1.5) + np.sqrt(0.5))
    say("check: f(1.5) = %.5f (closed form %.5f), F(1.5) = %.5f, f(2) = %.5f, F(2) = %.5f, f(39) = %.6f, F(39) = %.6f"
        % (a[0], cl, a[1], *ff(2.0), *ff(39.0)))
    for (eta, th) in [(0.008, 0.39), (0.0401, 0.3455)]:
        say("J(%.4f, %.4f) at hw = 2e-3, 1e-3, 5e-4: %s" % (eta, th, [round(J(eta, th, hw), 5) for hw in (2e-3, 1e-3, 5e-4)]))

    say("\nUnweighted (L = 1): margin = main - 8 J, r = floor(1/eta)")
    for r in range(20, 27):
        eta = 1 / (r + 1) + 1e-4; best = None
        for th in np.arange(0.2505, 0.4999, 0.005):
            P, arg = Phi(eta, th)
            if P <= 0: continue
            M = c * P / np.sqrt(eta * th) - 8 * J(eta, th)
            if best is None or M > best[0]: best = (M, th, arg)
        say("  r = %2d  eta = %.4f  th = %.4f  s1 = %.3f  s3 = %.3f  margin = %+.4f" % (r, eta, best[1], *best[2], best[0]))

    setups = {1.0: ([0.004, 0.006, 0.008, 0.01, 0.012, 0.015, 0.02], range(6, 12)),
              1.5: ([0.003, 0.004, 0.005, 0.006, 0.008, 0.01], range(9, 15)),
              3.0: ([0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005], range(15, 22))}
    ths = np.arange(0.33, 0.4951, 0.005)
    curves = {}; olds = {}
    for L, (etas, rs) in setups.items():
        say("\nWeighted (Richert weights also on the bad survivors), switching loss L = %.1f" % L)
        curves[L] = []; olds[L] = []
        for r in rs:
            b, o = weighted(r, L, etas, ths)
            curves[L].append((r, b["M"], b["main"], b["T"])); olds[L].append((r, o["M"], o["main"], o["T"]))
            say("  r = %2d  margin = %+.4f (%.1f%%)  (main %.4f, switched %.4f)  eta = %.4f  th = %.3f  u0 = %.3f  1/lam = %.3f  s1 = %.3f  s3 = %.3f  | bad survivors at weight 1: %+.4f"
                % (r, b["M"], 100 * b["M"] / b["main"], b["main"], b["T"], b["eta"], b["th"], b["u0"], 1 / b["lam"],
                   b["s"][0], b["s"][1], o["M"]))
        rmin = min([r for r, M, _, _ in curves[L] if M > 0])
        rsafe = min([r for r, M, mn, _ in curves[L] if M > 0.05 * mn])
        rold = min([r for r, M, mn, _ in olds[L] if M > 0.05 * mn])
        say("  => L = %.1f: smallest r with margin > 0: %d; with margin > 5%% of main: %d (= r_0(L)); bad survivors at weight 1: %d"
            % (L, rmin, rsafe, rold))
    np.save(os.path.join(RES, "semilinear_curves.npy"), {"weighted": {str(k): v for k, v in curves.items()}, "weight1": {str(k): v for k, v in olds.items()}}, allow_pickle=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    cols = {1.0: "C0", 1.5: "C1", 3.0: "C3"}
    for L, data in curves.items():
        rr = [d[0] for d in data]; MM = [d[1] / d[2] for d in data]
        ax[0].plot(rr, MM, "o-", color=cols[L], label=r"$L=%g$" % L)
        ax[0].plot([d[0] for d in olds[L]], [d[1] / d[2] for d in olds[L]], ":", color=cols[L], lw=1)
        rmin = min([d[0] for d in data if d[1] > 0.05 * d[2]])
        off = {1.0: (-40, 8), 1.5: (10, -24), 3.0: (10, -24)}[L]
        ax[0].annotate(r"$r_0=%d$" % rmin, (rmin, [d[1] / d[2] for d in data if d[0] == rmin][0]),
                       textcoords="offset points", xytext=off, color=cols[L], fontsize=10)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel(r"$r$"); ax[0].set_ylabel(r"(main $-$ switched)/main")
    ax[0].plot([], [], "k:", lw=1, label="bad survivors at weight 1")
    ax[0].set_title("Richert-weighted margin"); ax[0].legend(frameon=False, fontsize=9)
    for L, data in curves.items():
        rr = [d[0] for d in data]
        ax[1].plot(rr, [d[2] for d in data], "-", color=cols[L])
        ax[1].plot(rr, [d[3] for d in data], "--", color=cols[L])
    ax[1].plot([], [], "k-", label="weighted main term"); ax[1].plot([], [], "k--", label=r"switched term $8LJ_\lambda$")
    ax[1].set_xlabel(r"$r$"); ax[1].set_ylabel(r"units of $2\mathcal{B}$")
    ax[1].set_title("main term and switched term at the optimum"); ax[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sieve_margin.pdf")); fig.savefig(os.path.join(FIG, "sieve_margin.png"), dpi=150)
    say("\nfigure written; total time %.0f s" % (time.time() - t0))


