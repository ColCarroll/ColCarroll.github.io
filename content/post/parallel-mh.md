+++
title = "Very parallel MCMC sampling"
date = 2019-08-18T12:36:15-04:00
draft = false
summary = "Four chains isn't cool. You know what's cool? A million chains."

# Tags and categories
# For example, use `tags = []` for no tags, or the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags = []
categories = []

# Featured image
# Place your image in the `static/img/` folder and reference its filename below, e.g. `image = "example.jpg"`.
# Use `caption` to display an image caption.
#   Markdown linking is allowed, e.g. `caption = "[Image credit](http://example.org)"`.
# Set `preview` to `false` to disable the thumbnail in listings.
[header]
image = ""
caption = ""
preview = true

+++

I have been spending a lot of time with [tensorflow probability](https://www.tensorflow.org/probability) in the last year in working on PyMC4 and generally doing *Bayesian things*. One feature that I have not seen emphasized, but I find very cool is that chains are practically free, meaning running hundreds or thousands of chains is about as expensive as running 1 or 4.

## What is a chain?

A *chain* [is an independent run of MCMC](https://stackoverflow.com/questions/49825216/what-is-a-chain-in-pymc3/49836257#49836257). Running multiple chains can help diagnose multimodality (as in the linked answer), and allows for [convergence diagnostics](https://avehtari.github.io/rhat_ess/rhat_ess.html). Chains also play a part in creating [shape problems in PyMC3](https://github.com/pymc-devs/pymc3/issues?utf8=%E2%9C%93&q=label%3Ashape_problem), and why we recommend [xarray](http://xarray.pydata.org/en/stable/) to store probabilistic programming data: 1,000 draws of a 3 dimensional random variable with 4 chains will (probably) have shape (1000, 4, 3).

{{< figure src="/img/mixture.png" caption="Sampling from a mixture of six Gaussians using four chains looks pretty fishy. The left is a histogram from each of the four chains, and the right is a timeseries of the 1,000 draws for each of the chains, where you can see the chains jumping from one mode to the next." >}}

## How are chains usually implemented?

I have always seen libraries

1. Implement an MCMC sampler, and then
2. Use some sort of multiprocessing library to repeat it multiple times.

For example, PyMC3 used to use [joblib](https://joblib.readthedocs.io/en/latest/), and now uses [a custom implementation](https://github.com/pymc-devs/pymc3/pull/3011). So if you have 4 cores, you will run 4 independent chains in about the same amount of time as a single chain, or 100 independent chains in ~25x the amount of time as a single chain.

{{< figure src="/img/sampling.png" caption="By default, PyMC3 will run one chain for each core available. This used 4 cores to sample 4 chains, and did it in less than a second." >}}

## What are we impressed by again?

If you can write an algorithm for "MCMC with multiple chains" as a vectorized routine, then instead of running your algorithm for "MCMC" multiple times, you can run "MCMC with multiple chains" once. Hopefully the linear algebra you used gives you performance gains, too.

In a very hand-wavy way, we go from

    usual_version = [run_mcmc(iters) for _ in range(chains)]

to

    vectorized_version = run_mcmc(iters, chains)

## What can we do with thousands of chains?

My impression is that there is low-hanging fruit here, but there are a couple places I have seen or found thousands of chains to be useful.

1. I have been playing with [unbiased MCMC with couplings](http://arxiv.org/abs/1708.03625) recently. What does it mean for MCMC to be unbiased? It turns out that with imperfect initialization, your MCMC is only *asymptotically* from the stationary distribution, so plotting the mean of thousands of (finite-length) chains will be biased:
![png](/img/biased_mcmc.png)

2. In the very interesting [NeuTra-Lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport]( http://arxiv.org/abs/1903.03704), the authors use thousands of chains in the experiments to report estimates of the algorithm's performance:
![png](/img/experiments.png)

3. It seems like there is something useful to be done with [tuning MCMC algorithms](https://colcarroll.github.io/hmc_tuning_talk/). For example, step size adaptation is a one dimensional stochastic optimization problem, and may be able to be "solved" with grid search: choose a heuristic upper and lower bound on the step size, run a few iterations with step size `tf.linspace(lower, upper, num_chains)`, and then choose the optimal step size.

## How can I use thousands of chains?

**TensorFlow Probability** Here is [a gist](https://gist.github.com/ColCarroll/17c7fb6da0b8e3a32996ffa3c8826d46) showing how to run Hamiltonian Monte Carlo in TensorFlow probability with 256 chains.

**Numpy** Here is a `numpy` implementation of a Metropolis-Hastings sampler, to give a taste of what it looks like internally:

    import numpy as np

    def metropolis_hastings_vec(log_prob, proposal_cov, iters, chains, init):
        """Vectorized Metropolis-Hastings.

        Allows pretty ridiculous scaling across chains:
        Runs 1,000 chains of 1,000 iterations each on a
        correlated 100D normal in ~5 seconds.
        """
        proposal_cov = np.atleast_2d(proposal_cov)
        dim = proposal_cov.shape[0]
        if init.shape == (dim,):
            init = np.tile(init, (chains, 1))

        samples = np.empty((iters, chains, dim))
        samples[0] = init
        current_log_prob = log_prob(init)

        proposals = np.random.multivariate_normal(np.zeros(dim), proposal_cov,
                                                  size=(iters - 1, chains))
        log_unifs = np.log(np.random.rand(iters - 1, chains))
        for idx, (sample, log_unif) in enumerate(zip(proposals, log_unifs), 1):
            proposal = sample + samples[idx - 1]
            proposal_log_prob = log_prob(proposal)
            accept = (log_unif < proposal_log_prob - current_log_prob)

            # copy previous row, update accepted indexes
            samples[idx] = samples[idx - 1]
            samples[idx][accept] = proposal[accept]

            # update log probability
            current_log_prob[accept] = proposal_log_prob[accept]
        return samples

You can use this sampler with, for example,

    import scipy.stats as st

    dim = 10
    Σ = 0.1 * np.eye(dim) + 0.9 * np.ones((dim, dim))

    # Correlated Gaussian
    log_prob = st.multivariate_normal(np.zeros(dim),  Σ).logpdf

    proposal_cov = np.eye(dim)
    iters = 2_000
    chains= 1_024
    init = np.zeros(dim)

    samples = metropolis_hastings_vec(log_prob, proposal_cov, iters, chains, init)
