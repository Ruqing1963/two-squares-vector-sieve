"""Convergence checks at the optima of Table 1 (margins with the Buchstab step of Lemma 4.3).
Three grid levels, each refining all of the following at once:
  the grid of cuts theta0 / inert primes (220, 440, 880 geometric points),
  the Richert t-grid (120, 240, 480 geometric points),
  the switched-term grid hw (1e-3, 5e-4, 2.5e-4; for eta < 0.006 the production grid is 5e-4).
Also: eta -> 0 check of the Buchstab identity, Phi_B -> f(1/(2 theta)) for every cut theta0."""
import numpy as np, importlib.util, os, sys
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("ss", os.path.join(os.path.dirname(os.path.abspath(__file__)), "semilinear_sieve.py"))
ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
out = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "check_richert_grid.txt"), "w", encoding="utf8")
def say(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()

th = 0.42
for th0 in (0.1, 0.2, 0.3):
    a = np.linspace(th0, th, 2001)
    g = np.array([ss.ff((0.5 - x) / x)[1] for x in a])
    val = np.sqrt(th / th0) * ss.ff(0.5 / th0)[0] - 0.5 * np.trapz(np.sqrt(th / a) * g / a, a)
    say("eta -> 0, theta = %.2f, theta0 = %.1f: Buchstab value %.5f, f(1/(2 theta)) = %.5f" % (th, th0, val, ss.ff(0.5 / th)[0]))

OPT = [(0.008, 0.370, 0.335, 8, 1.0), (0.005, 0.420, 0.286, 11, 1.5), (0.0025, 0.465, 0.225, 17, 3.0), (0.002, 0.465, 0.219, 18, 3.0)]
for (eta, th, u0, r, L) in OPT:
    lam = 1 / (r + 1 - 1e-6 - 1 / u0)
    row = []
    for nA, nT, hw in ((220, 120, 1e-3), (440, 240, 5e-4), (880, 480, 2.5e-4)):
        if eta < 0.006 and hw > 5e-4: continue
        ss.AG = np.geomspace(0.01, 0.495, nA); ss.btab.clear()
        ts = np.geomspace(eta, 0.49, nT)
        PB, GB, th0 = ss.buchstab(eta, th, ts)
        m = ts <= u0
        integ = np.trapz((1 - ts[m] / u0) * GB[m] / ts[m], ts[m])
        main = ss.c * (PB - lam / 2 * integ) / np.sqrt(eta * th)
        T = 8 * L * ss.Jlam(ss.Jparts(eta, th, hw), lam, u0)
        row.append("(%d,%d,%.1e): main %.4f switched %.4f M %+.4f theta0 %.3f" % (nA, nT, hw, main, T, main - T, th0))
    say("eta=%.4f th=%.3f u0=%.3f r=%d L=%.1f\n   " % (eta, th, u0, r, L) + "\n   ".join(row))
