# OptQuantMMD
This repository contains code pertaining to the article "Optimal Quantisation of Probabililty Measures Using Maximum Mean Discrepancy" by Teymur et al. (AISTATS 2021)

The dataset linked here https://doi.org/10.7910/DVN/MDKNWM and created by the authors of "Optimal Thinning of MCMC Output" by Riabiz et al. (JRSSB 2021)

Our implementation of these samplers interfaces with the CVODES library (Hindmarsh et al., 2005), which presents a practical barrier to reproducibility. Moreover, the CPU time required to obtain MCMC samples was approximately two weeks for the calcium model. Since our research focused on post-processing of MCMC output, rather than MCMC itself, we directly make available the full output from each sampler on each model considered at

https://doi.org/10.7910/DVN/MDKNWM

This Harvard database download link consists of a single ZIP archive (1.5GB) that contains, for each ODE model and each MCMC method, the states (xi)ni=1 visited by the Markov chain, their corresponding gradients ∇logp(xi) and the values p(xi) up to an unknown normalisation constant. The Stein Thinning software described in S1 can be used to post- process these datasets at minimal effort, enabling our findings to be reproduced.