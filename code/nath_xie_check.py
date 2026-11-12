"""Diagnosis of the numerical value H(0.17, 0.25, 0.449, 0.011) = 1.2690 of Nath-Xie (arXiv:2501.16723, Section 5).

Their Matlab files (Computation_files/lower_f_sigma.m, upper_F_sigma.m, bundled in the arXiv source) integrate the
linear-sieve functions from the delay system  F' = f(s-1) - F,  f' = F(s-1) - f  for s > 2, with history F = 2e^gamma/s,
f = 0 on [1, 2].  The linear sieve functions satisfy (sF)' = f(s-1), (sf)' = F(s-1), with F = 2e^gamma/s on (0, 3] and
f = 2e^gamma log(s-1)/s on [2, 4]; the code's system lacks the factor 1/s and the initial segments, so its "f_2" exceeds
its "F_2" (e.g. at s = 3: 1.48 against 0.65).  Here H is assembled exactly as in H.m with (a) the code's functions and
(b) the correct ones, at their point, together with the pieces.  Output: results/nath_xie_check.txt"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
out = open(os.path.join(HERE, "..", "results", "nath_xie_check.txt"), "w", encoding="utf8")
def say(s=""):
    print(s, flush=True); out.write(s + "\n")
g = 0.5772156649; eg = np.exp(g); A = 2 * np.sqrt(eg / np.pi)

# ---- constants exactly as in the .m files (prime bounds 1e6 / 1e4; c3 truncated at 1e4 as coded) ----
def primes(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return np.nonzero(s)[0].astype(float)
P6 = primes(10 ** 6); P4 = primes(10 ** 4)
p3 = P6[(P6 % 4 == 3) & (P6 > 3)]; p1 = P6[P6 % 4 == 1]
C = 2.25 * np.exp(np.sum(np.log(1 - (3 * p3 - 1) / (p3 - 1) ** 3)) + np.sum(np.log(1 - 1 / (p1 - 1) ** 2)))
kappa = np.sqrt(np.pi / 2 * np.prod(1 - 1 / P6[P6 % 4 == 3] ** 2))
c1 = np.prod(1 + 1 / (P4[P4 > 3] - 2) ** 2); c2 = np.prod(1 - 1 / (P4[P4 > 2] - 1) ** 2)
q1 = P4[P4 % 4 == 1]; q3 = P4[P4 % 4 == 3]
c3_code = 2 / np.sqrt(np.pi) * np.prod((1 - 1 / q1) ** 0.5 * (1 + q1 ** 2 / (q1 - 1) ** 3)) * np.prod((1 - 1 / q3) ** 0.5)
c3_true = np.prod((1 - 1 / p1) * (1 + p1 ** 2 / (p1 - 1) ** 3)) * np.sqrt(np.prod(1 - 1 / P6[P6 % 4 == 3] ** 2))
pref = C * kappa / (4 * np.exp(1.5 * g))
say("constants as coded: C = %.6f kappa = %.6f c1 = %.6f c2 = %.6f c3 (truncated at 1e4, as coded) = %.6f; c3 in convergent form = %.6f; prefactor %.6f"
    % (C, kappa, c1, c2, c3_code, c3_true, pref))

# ---- sieve functions on a grid, step 1e-3 ----
hh = 1e-3; S = np.arange(hh, 60 + hh / 2, hh); n = len(S); lag = int(round(1 / hh))
def idx(s): return np.clip(np.round(np.minimum(np.asarray(s, float), 59.9) / hh).astype(int) - 1, 0, n - 1)
# half-dimensional (correct in the code): F = A/sqrt(s) on (0,2], f = A/(2 sqrt s) log(1+2(s-1)+2 sqrt(s(s-1))) on (1,2], then the DDE
F1 = np.where(S <= 2, A / np.sqrt(S), 0.0); f1 = np.where((S > 1) & (S <= 2), A / (2 * np.sqrt(S)) * np.log(np.maximum(1 + 2 * (S - 1) + 2 * np.sqrt(np.maximum(S * (S - 1), 0)), 1)), 0.0)
i2 = int(round(2 / hh)) - 1
for i in range(i2 + 1, n):            # F' = (f(s-1) - F)/(2s), f' = (F(s-1) - f)/(2s)
    F1[i] = F1[i - 1] + hh * (f1[i - 1 - lag] - F1[i - 1]) / (2 * S[i - 1])
    f1[i] = f1[i - 1] + hh * (F1[i - 1 - lag] - f1[i - 1]) / (2 * S[i - 1])
# linear, as coded (wrong): history F = 2e^g/s, f = 0 on [1,2]; then F' = f(s-1) - F, f' = F(s-1) - f
Fw = np.where(S <= 2, 2 * eg / np.maximum(S, 1e-9), 0.0); fw = np.zeros(n)
for i in range(i2 + 1, n):
    Fw[i] = Fw[i - 1] + hh * (fw[i - 1 - lag] - Fw[i - 1])
    fw[i] = fw[i - 1] + hh * (Fw[i - 1 - lag] - fw[i - 1])
# linear, correct: F = 2e^g/s on (0,3], f = 2e^g log(s-1)/s on [2,4]; (sF)' = f(s-1), (sf)' = F(s-1)
Fc = np.where(S <= 3, 2 * eg / S, 0.0); fc = np.where((S >= 2) & (S <= 4), 2 * eg * np.log(np.maximum(S - 1, 1)) / S, 0.0)
sF = S * Fc; sf = S * fc; i3 = int(round(3 / hh)) - 1; i4 = int(round(4 / hh)) - 1
for i in range(i3 + 1, n):
    sF[i] = sF[i - 1] + hh * 0.5 * (fc[i - lag] + fc[i - 1 - lag]); Fc[i] = sF[i] / S[i]
    if i > i4: sf[i] = sf[i - 1] + hh * 0.5 * (Fc[i - lag] + Fc[i - 1 - lag]); fc[i] = sf[i] / S[i]
say("linear sieve at s = 3, 4, 5: as coded F2 = %.4f %.4f %.4f, f2 = %.4f %.4f %.4f;  correct F2 = %.4f %.4f %.4f, f2 = %.4f %.4f %.4f"
    % (*Fw[idx([3, 4, 5])], *fw[idx([3, 4, 5])], *Fc[idx([3, 4, 5])], *fc[idx([3, 4, 5])]))

def f_sig(s1g, s2g, F2, f2, m=20001):
    """sup of f1 F2 + f2 F1 - F1 F2 over s1/s1g + s2/s2g = 1, s1 >= 1, s2 >= 2 (grid)"""
    s1 = np.linspace(1.0, s1g * (1 - 2 / s2g), m); s2 = s2g * (1 - s1 / s1g)
    v = f1[idx(s1)] * F2[idx(s2)] + f2[idx(s2)] * F1[idx(s1)] - F1[idx(s1)] * F2[idx(s2)]
    k = int(np.argmax(v)); return v[k], s1[k], s2[k]
def F_sig(s1g, s2g, F2, m=20001):
    """inf of F1 F2 over s1/s1g + s2/s2g = 1, s1 >= 1, s2 >= 2 (as coded; the paper allows s1 > 0, s2 >= 1)"""
    if s1g * (1 - 2 / s2g) < 1: return np.inf
    s1 = np.linspace(1.0, s1g * (1 - 2 / s2g), m); s2 = s2g * (1 - s1 / s1g)
    return (F1[idx(s1)] * F2[idx(s2)]).min()
def Cth_code(t1):
    b = np.linspace(0, np.sqrt(1 - 2 * t1), 20001)[1:] ** 2
    return np.trapz(2 * np.log((1 - b - t1) / t1) / np.sqrt(1 - b), np.sqrt(b))      # code: (1-b)^{-1/2}
def Cth_paper(t1):
    b = np.linspace(0, np.sqrt(1 - 2 * t1), 20001)[1:] ** 2
    return np.trapz(2 * np.log((1 - b - t1) / t1) / (1 - b), np.sqrt(b))              # paper: (1-b)^{-1}
def I_int(th, t1, t2, F2):
    al = np.geomspace(t2, th, 600)
    vals = np.array([F_sig((1 - 2 * a) / (2 * t1), (1 - 2 * a) / (2 * t2), F2, 2001) for a in al])
    ok = np.isfinite(vals)
    return np.trapz(((1 - al / th) * vals / al)[ok], al[ok])

lam, th, t1, t2 = 0.17, 0.25, 0.449, 0.011
for label, F2, f2, c3, Cf in (("code's functions, c3 and C(theta1) as coded", Fw, fw, c3_code, Cth_code),
                              ("correct linear sieve, same constants", Fc, fc, c3_code, Cth_code),
                              ("correct linear sieve, convergent c3, C(theta1) as in the paper", Fc, fc, c3_true, Cth_paper)):
    fv, s1o, s2o = f_sig(1 / (2 * t1), 1 / (2 * t2), F2, f2)
    I = I_int(th, t1, t2, F2); Ct = Cf(t1)
    main = pref * fv / (np.sqrt(t1) * t2); T = 2 * c1 * c2 ** 2 * c3 * Ct / (t2 * min(t1, 0.5 - t2)); rich = lam * pref * I / (np.sqrt(t1) * t2)
    say("\n%s:\n  f(sigma1, sigma2) = %.5f at (s1, s2) = (%.4f, %.4f); I = %.5f; C(theta1) = %.5f\n  main %.4f  switched %.4f  lambda*Richert %.4f  =>  H = %.4f   (paper: 1.2690)"
        % (label, fv, s1o, s2o, I, Ct, main, T, rich, main - T - rich))
# the unweighted search point of the paper
t1, t2 = 0.4230, 0.0249
for label, F2, f2 in (("code's functions", Fw, fw), ("correct", Fc, fc)):
    fv = f_sig(1 / (2 * t1), 1 / (2 * t2), F2, f2)[0]
    G = pref * fv / (np.sqrt(t1) * t2) - 2 * c1 * c2 ** 2 * c3_code * Cth_code(t1) / (t2 * min(t1, 0.5 - t2))
    say("\nunweighted G(0.4230, 0.0249) with %s: f = %.5f, G = %.6f (paper: 0.001378)" % (label, fv, G))
