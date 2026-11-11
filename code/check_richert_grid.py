"""Convergence of the Richert integral int_eta^u0 (1 - t/u0) G(t) dt/t in the grid of t (geometric grids)."""
import numpy as np, importlib.util, os, sys
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("ss", os.path.join(os.path.dirname(os.path.abspath(__file__)), "semilinear_sieve.py"))
ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
for (eta, th, u0, r, L) in [(0.008, 0.395, 0.335, 9, 1.0), (0.005, 0.43, 0.286, 12, 1.5), (0.0025, 0.465, 0.220, 18, 3.0), (0.002, 0.465, 0.219, 19, 3.0)]:
    P, arg = ss.Phi(eta, th)
    jj = ss.J(eta, th, hw=5e-4)
    lam = 1 / (r + 1 - 1e-6 - 1 / u0)
    vals = []
    for n in (120, 240, 480, 960):
        ts = np.geomspace(eta, u0, n)
        Gs = np.array([ss.G(t, eta, th) for t in ts])
        integ = np.trapz((1 - ts / u0) * Gs / ts, ts)
        main = ss.c * (P - lam / 2 * integ) / np.sqrt(eta * th)
        vals.append(round(main - 8 * L * jj, 4))
    print("eta=%.4f th=%.3f u0=%.3f r=%d L=%.1f  J(5e-4)=%.5f  margin for n = 120, 240, 480, 960: %s" % (eta, th, u0, r, L, jj, vals))
