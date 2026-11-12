# Sums of two squares and Heegner norms with few prime factors, XIII

**Half-dimensional vector sieves, class-group covering, and explicit small r for arbitrary class pairs (C', C)**

Ruqing Chen (GUT Geoservice Inc., Montreal)

This paper replaces the weighted Titchmarsh-type count of Parts X–XII by Iwaniec's half-dimensional sieve. Only the
Bombieri–Vinogradov level N^{1/2} for the primes is needed. Main results:

- N = Q(u,v)_prime + Q(x,y)_{P_7} for the nine discriminants of class number one (large even admissible N).
- For D = 23 (h = 3), every class pair (C', C) gets an explicit r(C', C) in {8, 9, 11, 12, 17, 18}.

Contents:

- `paper/two_squares_part13.tex`: the paper.
  - Theorem 1.1: Bombieri–Vinogradov with rational moduli up to x^{1/2}(log x)^{-B}, for the primes represented by one form.
  - Lemma 2.1: covering in the class group.
  - Lemma 4.1 and Proposition 4.2: the normalised sieve criterion.
  - Lemma 4.3: iterated Buchstab steps on the inert primes (a recursion Phi_m, G_m; its fixed point lowers r_0 from 8, 11, 18 to 7, 10, 16).
  - Theorems 1.2–1.4 and Corollary 1.5: the explicit values of r.
    - h = 1 gives P_7.
    - Idoneal D give P_7 (or P_{7+a}).
    - D = 23 gives the table.
    - Every prime D = 3 mod 4 gives r <= 125 eps h + h - 1.
  - Proposition 4.4: every switching loss L is absorbed, with r_0(L) << L log 2L.
- `code/semilinear_sieve.py`: semi-linear sieve functions, vector sieve, switching, Richert weights, one Buchstab step; computes
  the one-step thresholds 8, 11, 17 for the losses L = 1, 3/2, 3 (8, 11, 18 with the rectangle bounds) and calls the figure script (about 13 minutes).
- `code/iterated_buchstab.py`: the recursion of Lemma 4.3 by 2-D dynamic programming (uniform grid, m up to 12); eta -> 0 self-check,
  margins at the fixed point, grid refinement n = 400, 800, 1600, one-sided (safe) cell rule; gives r_0(L) = 7, 10, 16.
  Modes: `python iterated_buchstab.py` (all, about 10 minutes), `fine`, `safe`, `safe1600` (one-sided rule on n = 1600, about 25 minutes), `curve` (Figure 1 data).
- `code/sieve_margin_figure.py`: draws Figure 1 from the saved curves of the two scripts above.
- `code/check_richert_grid.py`: grid convergence of the one-step margins (cuts, Richert integral, J_lambda) at the optima, and the eta -> 0 check of the one-step identity.
- `code/nano_sieve_probe.py`: probe of Buchstab steps on the inert primes and on the split primes (the latter gains nothing).
- `code/prop43.py`: checks Proposition 4.4 (explicit r_0(L) for every L).
- `code/d23_classes.py`: D = 23 checks of the covering lemma (all m <= 1.1e6) and class-pair counts, and Figure 2.
- `results/`, `figures/`: outputs.

Run with `python <script>.py` (numpy, matplotlib).

Repository: https://github.com/Ruqing1963/two-squares-vector-sieve. DOI: [10.5281/zenodo.24125302](https://doi.org/10.5281/zenodo.24125302).

Earlier parts: Part X doi:10.5281/zenodo.24070749, Part XI doi:10.5281/zenodo.24082184, Part XII doi:10.5281/zenodo.24084767.

Draft, not refereed. The sieve inequality is optimized numerically at first order (see Section 7 of the paper).
