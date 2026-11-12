"""Figure 1 (figures/sieve_margin.pdf): the relative margin M/main against r for L = 1, 3/2, 3.
Reads results/semilinear_curves.npy (semilinear_sieve.py: one Buchstab step, rectangle bounds, bad survivors at weight 1)
and results/iterated_buchstab_curves.npy (iterated_buchstab.py curve: the fixed point m = 12).
Left: fixed point (solid, markers), one step (dash-dot), rectangle (dashed), rectangle with weight 1 (dotted).
Right: weighted main term and switched term 8 L J_lambda at the fixed-point optimum."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); FIG = os.path.join(HERE, "..", "figures")

def draw():
    old = np.load(os.path.join(RES, "semilinear_curves.npy"), allow_pickle=True).item()
    fp_path = os.path.join(RES, "iterated_buchstab_curves.npy")
    fixed = np.load(fp_path, allow_pickle=True).item() if os.path.exists(fp_path) else None
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    cols = {"1.0": "C0", "1.5": "C1", "3.0": "C3"}
    off = {"1.0": (-14, 12), "1.5": (12, -18), "3.0": (12, -18)}
    rel = lambda data: ([d[0] for d in data], [d[1] / d[2] for d in data])
    for L in ("1.0", "1.5", "3.0"):
        c = cols[L]
        main_curve = fixed[L] if fixed else old["buchstab"][L]
        ax[0].plot(*rel(main_curve), "o-", color=c, label=r"$L=%g$" % float(L))
        if fixed:
            ax[0].plot(*rel(old["buchstab"][L]), "-.", color=c, lw=1)
        ax[0].plot(*rel(old["rectangle"][L]), "--", color=c, lw=1)
        ax[0].plot(*rel(old["weight1"][L]), ":", color=c, lw=1)
        # annotate the certified r_0 (one-sided rule, n = 1600; Table 1), not the least r with M > 0 on this grid:
        # for L = 3/2 the point r = 9 is +0.004 with the trapezoid rule at n = 400, within the quadrature error
        rmin = {"1.0": 7, "1.5": 10, "3.0": 16}[L] if fixed else min(d[0] for d in main_curve if d[1] > 0)
        y0 = [d[1] / d[2] for d in main_curve if d[0] == rmin][0]
        ax[0].annotate(r"$r_0=%d$" % rmin, (rmin, y0), textcoords="offset points", xytext=off[L], color=c, fontsize=10)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel(r"$r$"); ax[0].set_ylabel(r"(main $-$ switched)/main")
    if fixed:
        ax[0].plot([], [], "k-", marker="o", label="iterated Buchstab (fixed point)")
        ax[0].plot([], [], "k-.", lw=1, label="one Buchstab step")
    ax[0].plot([], [], "k--", lw=1, label="rectangle bounds")
    ax[0].plot([], [], "k:", lw=1, label="rectangle, bad survivors at weight 1")
    ax[0].set_title("Richert-weighted margin"); ax[0].legend(frameon=False, fontsize=8)
    for L in ("1.0", "1.5", "3.0"):
        data = fixed[L] if fixed else old["buchstab"][L]
        rr = [d[0] for d in data]
        ax[1].plot(rr, [d[2] for d in data], "-", color=cols[L])
        ax[1].plot(rr, [d[3] for d in data], "--", color=cols[L])
    ax[1].plot([], [], "k-", label="weighted main term"); ax[1].plot([], [], "k--", label=r"switched term $8LJ_\lambda$")
    ax[1].set_xlabel(r"$r$"); ax[1].set_ylabel(r"units of $2\mathcal{B}$")
    ax[1].set_title("main term and switched term at the optimum"); ax[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sieve_margin.pdf")); fig.savefig(os.path.join(FIG, "sieve_margin.png"), dpi=150)

if __name__ == "__main__":
    draw()
