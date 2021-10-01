# OptQuantMMD

This repository contains code pertaining to the article "Optimal Quantisation of Probabililty Measures Using Maximum Mean Discrepancy" by Teymur et al. (AISTATS 2021)

A test dataset of 1000 points (with gradients) is included in data.npz. The algorithm in main.py corresponds to Algorithm 3 in the article; the discrepancy used is KSD. To use MMD, the analytical form of the target is required. Mini-batching and simultaneous selection are possible by varying arguments to the main function call.

The code uses commercial optimisation software that requires additional licences. This came about as a result of exploring the available arsenal of software available at the time of writing, rather than because freely-available alternatives do not exist. For greedy optimisation we use Gurobi and for the semidefinitee relaxation we use Mosek. Both can be licensed free to academic users. Open-source Python-native opimisers are available and can easily be substituted at the appropriate part of the code. For small problems, the performance difference is negligible. For large problems, the commercial packages often perform significantly better.

Datasets used in Section 4 of the article are available for download at https://doi.org/10.7910/DVN/MDKNWM. These were originally created and uploaded by the authors of "Optimal Thinning of MCMC Output" by Riabiz et al. (JRSSB 2021)

