"""Probe A3 (Part XIV planning): can a Buchstab step on the switched sequence B_sw = {N - q_0^a n : n in E}
lower the constant F_lin(1) = 2 e^gamma of the switched term?

Normalised by X V(N^{1/2}) (linear sieve, V(N^c)/V(N^{1/2}) = 1/(2c)):
  upper bound with sifting limit z = N^{c0} and level N^{1/2}:   s0 F(s0) with s0 = 1/(2 c0);
  Buchstab:  S(A; N^{1/2}) = S(A; N^{c0}) - sum_{N^{c0} <= p < N^{1/2}} S(A_p; p),
  the subtracted terms have level N^{1/2}/p and sifting limit p, i.e. s = (1/2 - c)/c, p = N^c,
  and the linear lower bound f(s) is positive only for s > 2, i.e. c < 1/6.
So the bound is  B(c0) = s0 F(s0) - int_{c0}^{1/6} (1/(2c)) f((1/2 - c)/c) dc/c = s0 F(s0) - int_3^{s0} f(s - 1) ds.
By (s F(s))' = f(s - 1) this equals 1 . F(1) = 2 e^gamma for every c0: no gain from the identity alone.
This script checks the identity numerically and lists the level theta_sw > 1/2 that would be needed instead
(the upper bound at level N^theta is F(2 theta) = 2 e^gamma / (2 theta) for 2 theta <= 3)."""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
out = open(os.path.join(HERE, "..", "results", "switch_probe.txt"), "w", encoding="utf8")
def say(s):
    print(s); out.write(s + "\n")

g = np.euler_gamma; C = 2 * np.exp(g)
h = 1e-4
S = np.arange(h, 20 + h / 2, h); n = len(S); lag = int(round(1 / h))
F = np.where(S <= 3 + 1e-12, C / S, 0.0)
f = np.where((S >= 2 - 1e-12) & (S <= 4 + 1e-12), C * np.log(np.maximum(S - 1, 1)) / S, 0.0)
sF = S * F; sf = S * f
i3 = int(round(3 / h)) - 1; i4 = int(round(4 / h)) - 1
for i in range(i3 + 1, n):                       # (s F)' = f(s-1)
    sF[i] = sF[i - 1] + h * 0.5 * (f[i - lag] + f[i - 1 - lag]); F[i] = sF[i] / S[i]
    if i > i4:                                   # (s f)' = F(s-1)
        sf[i] = sf[i - 1] + h * 0.5 * (F[i - lag] + F[i - 1 - lag]); f[i] = sf[i] / S[i]
def ev(arr, s): return arr[min(max(int(round(s / h)) - 1, 0), n - 1)]

say("linear sieve: F(1) = %.5f = 2e^gamma = %.5f; F(2) = %.5f, f(2) = %.5f, F(4) = %.5f, f(4) = %.5f, F(10) = %.5f, f(10) = %.5f"
    % (ev(F, 1), C, ev(F, 2), ev(f, 2), ev(F, 4), ev(f, 4), ev(F, 10), ev(f, 10)))
say("\nA3: Buchstab step on B_sw with sifting limit N^{c0}; bound B(c0) = s0 F(s0) - int_3^{s0} f(s-1) ds, s0 = 1/(2 c0)")
for c0 in (0.5, 0.25, 1 / 6, 0.125, 0.1, 0.08, 0.06, 0.05):
    s0 = 1 / (2 * c0)
    ss = np.arange(3, s0 + 1e-12, h) if s0 > 3 else np.array([3.0])
    sub = np.trapz([ev(f, s - 1) for s in ss], ss) if s0 > 3 else 0.0
    say("  c0 = %.4f  s0 = %5.2f   s0 F(s0) = %.5f   subtracted = %.5f   B(c0) = %.5f   B/(2e^gamma) = %.5f" % (c0, s0, s0 * ev(F, s0), sub, s0 * ev(F, s0) - sub, (s0 * ev(F, s0) - sub) / C))

say("\nLevel theta_sw > 1/2 needed for the failing r (one-sided values of Part XIII, Remark 5.1): factor F(2 theta)/F(1) = 1/(2 theta)")
for (L, r, need) in ((1.0, 6, 0.094), (1.5, 9, 0.019), (3.0, 15, 0.048)):
    say("  L = %.1f, r = %2d: switched term must drop by %.1f%%  =>  theta_sw >= %.4f" % (L, r, 100 * need, 1 / (2 * (1 - need))))
say("\nFor comparison: the BFI level 4/7 would give the factor 7/8 = 0.875, a drop of 12.5%.")
