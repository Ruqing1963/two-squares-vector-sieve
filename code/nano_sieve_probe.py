"""Probe of the "adaptive Buchstab" proposals (not used in the paper).

Probe 1: one Buchstab step on the inert primes P_3 at a cut N^{theta0}:
  main:    Phi_B1 = max_{theta0} [ sqrt(th/theta0) Phi(1/2; eta, theta0)
                                   - 1/2 int_{theta0}^{th} sqrt(th/a) G(1/2 - a; eta, a) da/a ]
  Richert: G_B1(t) = min_{theta0} [ sqrt(th/theta0) G(R; eta, theta0)
                                   - 1/2 int_{theta0}^{min(th, R/2)} sqrt(th/a) Phi+(R - a; eta, a) da/a ],  R = 1/2 - t.
Probe 2: one Buchstab step on the split primes P_1 at a cut N^{eta0}:
  main:    Phi_B2 = max_{eta0} [ sqrt(eta/eta0) Phi(1/2; eta0, th) - 1/2 int_{eta0}^{eta} sqrt(eta/u) G(1/2 - u; u, th) du/u ]
  Richert: G_B2(t) = min_{eta0} [ sqrt(eta/eta0) G(R; eta0, th) - 1/2 int_{eta0}^{eta} sqrt(eta/u) Phi+(R - u; u, th) du/u ].
Here Phi(X; e, a) = max_{e s1 + a s3 = X} [f1 F3 + F1 f3 - F1 F3] and G(X; e, a) = min_{e s1 + a s3 = X} F1 F3
(semi-linear f, F of semilinear_sieve.py); the 1/2 is the density of inert (resp. split) primes and the square
roots are the ratios V(z)/V(z') of dimension 1/2.
"Both" uses max(Phi_B1, Phi_B2) and min(G_B1, G_B2), each a valid bound.
"Ideal" uses Phi = f(1/(2 th)) and G(t) = F((1/2 - t)/th): split primes sifted at no cost of level.
No Buchstab iteration with these f, F can beat it (it is the value in the model where the two sieves are independent).
The switched term 8 L J_lambda is unchanged.
"""
import numpy as np, importlib.util, os, sys, time
sys.argv = ["x"]
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ss", os.path.join(HERE, "semilinear_sieve.py"))
ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
f12, F12, hS, c = ss.f12, ss.F12, ss.h, ss.c
n12 = len(f12)

out = open(os.path.join(HERE, "..", "results", "nano_sieve_probe.txt"), "w", encoding="utf8")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n"); out.flush()

def lk(arr, s):
    return arr[np.clip(np.round(np.minimum(s, 39.9) / hS).astype(int) - 1, 0, n12 - 1)]

def PhiX(X, e, a):
    """lower vector sieve at level N^X, split primes < N^e, inert primes < N^a (clipped at 0)"""
    smax = (X - a) / e
    if smax <= 1.0: return 0.0
    s1 = np.geomspace(1.0, smax, 500); s3 = (X - e * s1) / a
    v = lk(f12, s1) * lk(F12, s3) + lk(F12, s1) * lk(f12, s3) - lk(F12, s1) * lk(F12, s3)
    return max(0.0, v.max())

def GX(X, e, a):
    s1 = np.geomspace(1e-4, X / e * (1 - 1e-6), 800); s3 = (X - e * s1) / a
    k = s3 >= 1e-4
    return (lk(F12, s1[k]) * lk(F12, s3[k])).min()

def tsg(eta):                                 # Richert t-grid, as in semilinear_sieve.py
    return np.geomspace(eta, 0.49, 240)
AG = np.geomspace(0.01, 0.495, 440)           # cut / inert-prime grid for probe 1 (log grid)

def trap_cum_from(x, y):
    """C[i] = int_{x_i}^{x_end} y"""
    seg = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])

cache1 = {}
def probe1_tables(eta):
    """theta-independent pieces of probe 1 for this eta"""
    if eta in cache1: return cache1[eta]
    TS = tsg(eta)
    Pm = np.array([PhiX(0.5, eta, a) for a in AG])                # Phi(1/2; eta, theta0)
    Gm = np.array([GX(0.5 - a, eta, a) for a in AG])             # G(1/2 - a; eta, a)
    GR = np.array([[GX(0.5 - t, eta, a) for a in AG] for t in TS])            # G(R; eta, theta0)
    PR = np.array([[PhiX(0.5 - t - a, eta, a) if a < (0.5 - t) / 2 else 0.0 for a in AG] for t in TS])
    cache1[eta] = (Pm, Gm, GR, PR)
    return cache1[eta]

def probe1(eta, th):
    Pm, Gm, GR, PR = probe1_tables(eta); TS = tsg(eta)
    k = AG < th - 1e-9
    a = np.append(AG[k], th)          # the integrals must run up to theta exactly
    # main: candidates theta0 = a[i]; the last one (theta0 = theta) is the rectangle itself
    P0 = PhiX(0.5, eta, th)
    cum = trap_cum_from(np.log(a), np.append(Gm[k], GX(0.5 - th, eta, th)) * a ** -0.5)
    PhiB = max(P0, np.max(np.sqrt(th) * (np.append(Pm[k], P0) / np.sqrt(a) - 0.5 * cum)))
    GB = np.empty(len(TS))
    for i, t in enumerate(TS):
        R = 0.5 - t
        g0 = GX(R, eta, th)
        pr = np.append(PR[i, k], PhiX(R - th, eta, th) if th < R / 2 else 0.0)
        cum = trap_cum_from(np.log(a), pr * a ** -0.5)
        GB[i] = min(g0, np.min(np.sqrt(th) * (np.append(GR[i, k], g0) / np.sqrt(a) - 0.5 * cum)))
    return PhiB, GB

def probe2(eta, th):
    U = np.geomspace(eta / 30, eta, 40); TS = tsg(eta)
    Pm = np.array([PhiX(0.5, u, th) for u in U])
    Gm = np.array([GX(0.5 - u, u, th) for u in U])
    cum = trap_cum_from(np.log(U), Gm * U ** -0.5)
    PhiB = max(np.max(np.sqrt(eta) * (Pm / np.sqrt(U) - 0.5 * cum)), PhiX(0.5, eta, th))
    GB = np.empty(len(TS))
    for i, t in enumerate(TS):
        R = 0.5 - t
        g0 = GX(0.5 - t, eta, th)
        GRu = np.array([GX(R, u, th) for u in U])
        PRu = np.array([PhiX(R - u, u, th) for u in U])
        cum = trap_cum_from(np.log(U), PRu * U ** -0.5)
        GB[i] = min(g0, np.min(np.sqrt(eta) * (GRu / np.sqrt(U) - 0.5 * cum)))
    return PhiB, GB

VARS = ("rect", "fine", "B1", "B2", "both", "ideal")
cache = {}
def ingr(eta, th):
    key = (eta, round(th, 4)); TS = tsg(eta)
    if key not in cache:
        P, arg = ss.Phi(eta, th)
        d = dict(P={"rect": P}, G={})
        if P > 0:
            d["parts"] = ss.Jparts(eta, th, 5e-4 if eta < 0.006 else 1e-3)
            d["G"]["rect"] = np.array([ss.G(t, eta, th) for t in TS])
            P1, G1 = probe1(eta, th); P2, G2 = probe2(eta, th)
            d["P"]["fine"] = PhiX(0.5, eta, th); d["G"]["fine"] = np.array([GX(0.5 - t, eta, th) for t in TS])
            d["P"].update(B1=P1, B2=P2, both=max(P1, P2), ideal=ss.ff(0.5 / th)[0])
            d["G"].update(B1=G1, B2=G2, both=np.minimum(G1, G2), ideal=np.array([ss.ff((0.5 - t) / th)[1] for t in TS]))
        cache[key] = d
    return cache[key]

def best(r, L, etas, ths, var):
    b = None
    for eta in etas:
        for th in ths:
            d = ingr(eta, th); TS = tsg(eta)
            if d["P"]["rect"] <= 0: continue
            P, G, parts = d["P"][var], d["G"][var], d["parts"]
            j0 = ss.Jlam(parts, 0.0, 1.0)
            for u0 in np.linspace(eta + 0.01, 0.48, 300):
                il = r + 1 - 1e-6 - 1 / u0
                if il <= 0: continue
                lam = 1 / il; m = (TS >= eta) & (TS <= u0)
                if m.sum() < 2: continue
                integ = np.trapz((1 - TS[m] / u0) * G[m] / TS[m], TS[m])
                main = c * (P - lam / 2 * integ) / np.sqrt(eta * th)
                jj = ss.Jlam(parts, lam, u0) if 1 - 2 * th <= u0 else j0
                M = main - 8 * L * jj
                if b is None or M > b["M"]:
                    b = dict(M=M, eta=eta, th=th, u0=u0, main=main, T=8 * L * jj)
    return b

if __name__ == "__main__":
    t0 = time.time()
    # sanity 1: eta -> 0, probe 1 reproduces f(1/(2 th)) for every cut
    th = 0.42; e = 1e-7
    for th0 in (0.1, 0.2, 0.3):
        a = np.linspace(th0, th, 2001)
        g = np.array([ss.ff((0.5 - x) / x)[1] for x in a])
        val = np.sqrt(th / th0) * ss.ff(0.5 / th0)[0] - 0.5 * np.trapz(np.sqrt(th / a) * g / a, a)
        say("sanity eta->0: theta0=%.2f  Buchstab value %.5f  vs f(1/(2 theta)) = %.5f" % (th0, val, ss.ff(0.5 / th)[0]))
    # main-term gains at the Table 1 optima
    say("\nmain term in units 2B: c/sqrt(eta th) * Phi")
    for eta, th in ((0.008, 0.370), (0.004, 0.420), (0.002, 0.465)):
        d = ingr(eta, th); k = c / np.sqrt(eta * th); TS = tsg(eta)
        say("eta %.3f th %.3f: rect %.4f  fine %+.4f  B1 %+.4f  B2 %+.4f  ideal %+.4f   (gains over rect)" % (
            eta, th, k * d["P"]["rect"], k * (d["P"]["fine"] - d["P"]["rect"]), k * (d["P"]["B1"] - d["P"]["rect"]), k * (d["P"]["B2"] - d["P"]["rect"]),
            k * (d["P"]["ideal"] - d["P"]["rect"])))
        for t in (0.01, 0.05, 0.1, 0.2):
            i = np.argmin(abs(TS - t))
            say("     G at t=%.3f: rect %.4f  fine %.4f  B1 %.4f  B2 %.4f  ideal %.4f" % (TS[i], *(d["G"][v][i] for v in ("rect", "fine", "B1", "B2", "ideal"))))
    setups = {1.0: ([0.004, 0.006, 0.008, 0.01, 0.012], np.arange(0.34, 0.4251, 0.005), [7, 8]),
              1.5: ([0.003, 0.004, 0.005, 0.006, 0.008], np.arange(0.38, 0.4651, 0.005), [10, 11]),
              3.0: ([0.0015, 0.002, 0.0025, 0.003, 0.004], np.arange(0.42, 0.4951, 0.005), [16, 17, 18])}
    say("\nmargin M (percent of main); last column: cut of 2e^gamma in the switched term needed for M = 0")
    for L, (etas, ths, rs) in setups.items():
        for r in rs:
            line = "L=%.1f r=%2d:" % (L, r)
            for var in VARS:
                b = best(r, L, etas, ths, var)
                line += "  %s %+.3f (%+.1f%%)" % (var, b["M"], 100 * b["M"] / b["main"])
                if var == "both": need = max(0.0, 1 - b["main"] / b["T"])
            line += "   need %.1f%%" % (100 * need)
            say(line)
    say("time %.0f s" % (time.time() - t0))
