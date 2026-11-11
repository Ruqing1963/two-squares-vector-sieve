# Sums of two squares and Heegner norms with few prime factors, XIII

**Half-dimensional vector sieves, class-group covering, and explicit small r for arbitrary class pairs (C', C)**

Ruqing Chen (GUT Geoservice Inc., Montreal)

This paper replaces the weighted Titchmarsh-type count of Parts X–XII by Iwaniec's half-dimensional sieve. Only the
Bombieri–Vinogradov level N^{1/2} for the primes is needed. Main results:

- N = Q(u,v)_prime + Q(x,y)_{P_9} for the nine discriminants of class number one (large even admissible N).
- For D = 23 (h = 3), every class pair (C', C) gets an explicit r(C', C) in {10, 11, 13, 14, 20, 21}.

Contents:

- `paper/two_squares_part13.tex`: the paper.
  - Theorem 1.1: Bombieri–Vinogradov with rational moduli up to x^{1/2}(log x)^{-B}, for the primes represented by one form.
  - Lemma 2.1: covering in the class group.
  - Lemma 4.1 and Proposition 4.2: the normalised sieve criterion.
  - Theorems 1.2–1.4 and Corollary 1.5: the explicit values of r.
    - h = 1 gives P_9.
    - Idoneal D give P_9 (or P_{9+a}).
    - D = 23 gives the table.
    - Every prime D = 3 mod 4 gives r <= 125 eps h + h - 1.
  - Proposition 4.3: every switching loss L is absorbed, with r_0(L) << L log 2L.
- `code/semilinear_sieve.py`: semi-linear sieve functions, vector sieve, switching, Richert weights; computes
  r_0(L) = 9, 12, 19 for the losses L = 1, 3/2, 3, and draws Figure 1 (about 3 minutes).
- `code/check_richert_grid.py`: grid convergence of the Richert integral.
- `code/prop43.py`: checks Proposition 4.3 (explicit r_0(L) for every L).
- `code/d23_classes.py`: D = 23 checks of the covering lemma (all m <= 1.1e6) and class-pair counts, and Figure 2.
- `results/`, `figures/`: outputs.

Run with `python <script>.py` (numpy, matplotlib).

Earlier parts: Part X doi:10.5281/zenodo.24070749, Part XI doi:10.5281/zenodo.24082184, Part XII doi:10.5281/zenodo.24084767.

Draft, not refereed. The sieve inequality is optimized numerically at first order (see Section 7 of the paper).
