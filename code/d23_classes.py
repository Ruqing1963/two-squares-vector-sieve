"""Part XIII, D = 23 (h = 3): checks of the sumset covering lemma and a window of class-pair counts.
Q0 = x^2 + xy + 6y^2 (principal), Q1 = 2x^2 + xy + 3y^2 (= the classes C1, C2; same represented integers).
Run: python d23_classes.py   (a few minutes)
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); FIG = os.path.join(HERE, "..", "figures")
out = open(os.path.join(RES, "d23_classes.txt"), "w", encoding="utf8")
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n"); out.flush()

D = 23
NMAX = 1_100_000

def r_form(a, b, cc, nmax):
    """number of (x, y) in Z^2 with a x^2 + b x y + c y^2 = n, n <= nmax"""
    cnt = np.zeros(nmax + 1, dtype=np.int64)
    ymax = int(np.sqrt(4 * a * nmax / (4 * a * cc - b * b))) + 2
    for y in range(-ymax, ymax + 1):
        # a x^2 + b y x + c y^2 <= nmax
        disc = (b * y) ** 2 - 4 * a * (cc * y * y - nmax)
        if disc < 0: continue
        x0 = (-b * y - np.sqrt(disc)) / (2 * a); x1 = (-b * y + np.sqrt(disc)) / (2 * a)
        xs = np.arange(int(np.floor(x0)) - 1, int(np.ceil(x1)) + 2)
        v = a * xs * xs + b * xs * y + cc * y * y
        v = v[(v >= 0) & (v <= nmax)]
        cnt += np.bincount(v, minlength=nmax + 1)
    return cnt

r0 = r_form(1, 1, 6, NMAX)
r1 = r_form(2, 1, 3, NMAX)
say("r-arrays done; r0(1..10) =", r0[1:11].tolist(), " r1(1..10) =", r1[1:11].tolist())

# smallest prime factor sieve
spf = np.zeros(NMAX + 1, dtype=np.int64)
for p in range(2, int(NMAX ** 0.5) + 1):
    if spf[p] == 0:
        spf[p * p::p][spf[p * p::p] == 0] = p
idx = np.arange(NMAX + 1); spf[(spf == 0)] = idx[(spf == 0)]
isprime = (spf == idx) & (idx >= 2)

def kron(n):   # chi_{-23}(n) = (n/23) for n>0 (since -23 = 1 mod 4)
    n %= 23
    if n == 0: return 0
    return 1 if pow(n, 11, 23) == 1 else -1

# class of split primes: 0 = principal, 1 = non-principal
pcls = {}
primes = np.nonzero(isprime)[0]
for p in primes:
    if p == 23: continue
    if kron(int(p)) == 1:
        pcls[int(p)] = 0 if r0[p] > 0 else 1
say("check: 2 is", "principal" if pcls[2] == 0 else "non-principal", "; 3 is", "principal" if pcls[3] == 0 else "non-principal",
    "; 13 is", "principal" if pcls[13] == 0 else "non-principal")
fr = np.mean([v == 0 for k, v in pcls.items() if k > 1000])
say("fraction of principal among split primes in (1000, %d]: %.4f (expected 1/3)" % (NMAX, fr))

def factor(m):
    f = {}
    while m > 1:
        p = int(spf[m]); e = 0
        while m % p == 0: m //= p; e += 1
        f[p] = e
    return f

def predicted_classes(m):
    """set of classes (mod 3; 0 = C0, {1,2} = C1/C2) of ideals of norm m, or None if m is not a norm"""
    sset = {0}
    kn = 0
    for p, e in factor(m).items():
        if p == 23: continue
        k = kron(p)
        if k == -1:
            if e % 2: return None, None
            continue
        cl = pcls[p]
        if cl == 1: kn += e
        opts = {((e - 2 * j) * cl) % 3 for j in range(e + 1)}
        sset = {(a + b) % 3 for a in sset for b in opts}
    return sset, kn

fails = 0
for m in range(1, NMAX + 1):
    sset, kn = predicted_classes(m)
    pred0 = sset is not None and 0 in sset
    pred1 = sset is not None and (1 in sset or 2 in sset)
    if pred0 != (r0[m] > 0) or pred1 != (r1[m] > 0):
        fails += 1
say("sumset covering lemma: %d failures among 1 <= m <= %d" % (fails, NMAX))

Omega = np.zeros(NMAX + 1, dtype=np.int64)
for m in range(2, NMAX + 1):
    Omega[m] = Omega[m // spf[m]] + 1

# window of N
NS = [N for N in range(1_000_000, 1_000_400, 2) if N % 23]
rep_p = {0: (r0 > 0) & isprime, 1: (r1 > 0) & isprime}
W = {(a, b): [] for a in (0, 1) for b in (0, 1)}
kdist = np.zeros((4, 3), dtype=np.int64)    # rows Omega = 1, 2, 3, >=4 ; cols k_non = 0, 1, >=2
q0_ok = [0, 0]; q0_bad = [0, 0]
for N in NS:
    ps = np.nonzero(isprime[:N])[0]
    ms = N - ps
    for a in (0, 1):
        sel = rep_p[a][ps]
        mm = ms[sel]
        for b in (0, 1):
            rb = r0 if b == 0 else r1
            W[(a, b)].append(int(np.sum((rb[mm] > 0) & (Omega[mm] <= 3))))
    if N == NS[0] or N % 40 == 0:
        sel = (rep_p[0] | rep_p[1])[ps]
        for m in ms[sel]:
            m = int(m)
            sset, kn = predicted_classes(m)
            if sset is None: continue
            row = min(int(Omega[m]), 4) - 1
            kdist[row, min(kn, 2)] += 1
            if m % 3 == 0 and m % 9:
                q0_ok[0] += 1; q0_bad[0] += int(r1[m] == 0)
            if m % 9 == 0 and m % 27:
                q0_ok[1] += 1; q0_bad[1] += int(r0[m] == 0 or r1[m] == 0)
names = {0: "C0", 1: "C1"}
say("window: %d even N in [10^6, 10^6+400), 23 not dividing N" % len(NS))
for (a, b), v in W.items():
    say("  p in %s, N-p = Q_%s(x,y), Omega(N-p) <= 3: min %d, mean %.1f" % (names[a], names[b], min(v), np.mean(v)))
say("k_non distribution of represented m = N - p (rows Omega = 1,2,3,>=4; cols k_non = 0,1,>=2):")
for i in range(4): say("  ", kdist[i].tolist())
say("q0 = 3 checks (represented m in the sample): 3||m: %d cases, %d not represented by Q_C1; 9||m: %d cases, %d not represented by both"
    % (q0_ok[0], q0_bad[0], q0_ok[1], q0_bad[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "cm"})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
lab = [r"$\Omega=1$", r"$\Omega=2$", r"$\Omega=3$", r"$\Omega\geq4$"]
x = np.arange(4); wdt = 0.26
for j, (nm, col) in enumerate([(r"$k_{\rm non}=0$", "C2"), (r"$k_{\rm non}=1$", "C1"), (r"$k_{\rm non}\geq2$", "C0")]):
    ax[0].bar(x + (j - 1) * wdt, kdist[:, j], wdt, color=col, label=nm)
ax[0].set_xticks(x); ax[0].set_xticklabels(lab); ax[0].set_ylabel("number of $m=N-p$")
ax[0].set_title(r"$m=N-p$ represented by some form, $p\in C_0\cup C_1$"); ax[0].legend(frameon=False, fontsize=9)
for (a, b), v in W.items():
    ax[1].plot(np.array(NS) - 10**6, v, ".", ms=3, label=r"$p\in C_%d$, $N-p=Q_{C_%d}$" % (a, b))
ax[1].set_xlabel(r"$N-10^6$"); ax[1].set_ylabel(r"count with $\Omega(N-p)\leq3$")
ax[1].set_title(r"$D=23$: class pairs, $N\in[10^6,10^6+400)$ even"); ax[1].legend(frameon=False, fontsize=8, markerscale=3)
ax[1].ticklabel_format(axis="x", style="plain", useOffset=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "d23_classes.pdf")); fig.savefig(os.path.join(FIG, "d23_classes.png"), dpi=150)
say("figure written")

