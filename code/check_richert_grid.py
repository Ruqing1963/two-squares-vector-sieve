"""Convergence checks at the optima of Table 1:
the Richert integral int_eta^u0 (1 - t/u0) G(t) dt/t on geometric grids of 120..960 points, and the switched-term
integral J_lam on the grids hw = 1e-3, 5e-4, 2.5e-4."""
import numpy as np, importlib.util, os, sys
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("ss", os.path.join(os.path.dirname(os.path.abspath(__file__)), "semilinear_sieve.py"))
ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
for (eta, th, u0, r, L) in [(0.008, 0.370, 0.335, 8, 1.0), (0.004, 0.420, 0.302, 11, 1.5), (0.002, 0.465, 0.219, 18, 3.0)]:
    P, arg = ss.Phi(eta, th)
    lam = 1 / (r + 1 - 1e-6 - 1 / u0)
    Js = []
    for hw in (1e-3, 5e-4, 2.5e-4):
        Js.append(ss.Jlam(ss.Jparts(eta, th, hw), lam, u0))
    vals = []
    for n in (120, 240, 480, 960):
        ts = np.geomspace(eta, u0, n)
        Gs = np.array([ss.G(t, eta, th) for t in ts])
        integ = np.trapz((1 - ts / u0) * Gs / ts, ts)
        main = ss.c * (P - lam / 2 * integ) / np.sqrt(eta * th)
        vals.append(round(main - 8 * L * Js[-1], 4))
    print("eta=%.4f th=%.3f u0=%.3f r=%d L=%.1f  J_lam(hw=1e-3,5e-4,2.5e-4) = %s ; margin for n = 120,240,480,960: %s"
          % (eta, th, u0, r, L, [round(j, 5) for j in Js], vals))
