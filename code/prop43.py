"""Proposition 4.4: an unconditional explicit r_0(L) for every L (unweighted sieve, lambda = 0).
Choice: delta = 0.08/L, theta = (1-delta)/2, s3 = 1 + delta/2, s1 = s1(L) the least s >= 2 on the grid with
F(s) - f(s) <= 0.14 sqrt(delta), eta = delta(1+delta)/(4 s1).  Then r_0(L) <= floor(1/eta) < 50 L s1.
The script also evaluates the actual main term and J at these parameters, to check the chain of inequalities.
"""
import numpy as np, importlib.util, os, sys
sys.argv = ["x"]
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ss", os.path.join(here, "semilinear_sieve.py"))
ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
out = open(os.path.join(here, "..", "results", "prop43.txt"), "w", encoding="utf8")
def say(s): print(s); out.write(s + "\n")

S = np.arange(2.0, 39.0, 0.01)
gap = np.array([ss.ff(s)[1] - ss.ff(s)[0] for s in S])
Fmin = min(ss.ff(s)[1] for s in np.arange(0.05, 39.0, 0.01))
say("min of F(s) on (0,39]: %.6f ; F(1) = %.5f ; f(2) = %.5f" % (Fmin, ss.ff(1.0)[1], ss.ff(2.0)[0]))
say("F(s) - f(s): " + ", ".join("s=%g: %.2e" % (s, ss.ff(s)[1] - ss.ff(s)[0]) for s in (2, 2.5, 3, 3.5, 4, 5, 6, 8)))
say("%8s %10s %8s %10s %12s %12s %12s %10s" % ("L", "delta", "s1", "eta", "r bound", "main", "8LJ", "check"))
for L in [1, 1.5, 3, 10, 30, 100, 1000, 10**4, 10**5]:
    d = 0.08 / L
    th = (1 - d) / 2; s3 = 1 + d / 2
    ok = np.nonzero(gap <= 0.14 * np.sqrt(d))[0]
    s1 = S[ok[0]]
    eta = d * (1 + d) / (4 * s1)
    f1, F1 = ss.ff(s1)
    F3 = 2 * np.exp(ss.g / 2) / np.sqrt(np.pi * s3)
    f3 = np.exp(ss.g / 2) / np.sqrt(np.pi * s3) * 2 * np.log(np.sqrt(s3) + np.sqrt(s3 - 1))   # closed forms, s3 in [1,2]
    Phi = f1 * F3 + F1 * f3 - F1 * F3
    main = ss.c * Phi / np.sqrt(eta * th)
    hw = min(1e-3, d / 40)
    jj = ss.J(eta, th, hw=hw) if L <= 100 else float("nan")
    bound_main = 0.359 * np.sqrt(d) / np.sqrt(eta * th)
    bound_J = L * d ** 1.5 / (th ** 2 * np.sqrt(eta))
    say("%8g %10.2e %8.2f %10.3e %12d %12.4g %12.4g   main>=%.4g>%.4g>=8LJ  %s"
        % (L, d, s1, eta, int(1 / eta), main, 8 * L * jj, bound_main, bound_J,
           "OK" if (main >= bound_main > bound_J and (np.isnan(jj) or 8 * L * jj <= bound_J)) else "CHECK"))

