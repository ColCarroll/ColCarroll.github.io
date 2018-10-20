+++
date = "2018-10-20T10:06:15-04:00"
tags = []
draft = false
title = "Simulation Based Calibration in PyMC3"
highlight = true
math = false
summary = "An implementation of a recent paper from Talts, Betancourt, Simpson, Vehtari, and Gelman."
+++

![png](/img/fig10.png)
I got to see Sean Talts and Michael Betancourt giving very good (and crowded [[1](https://twitter.com/betanalpha/status/1052632528778084353)], [[2](https://twitter.com/betanalpha/status/1053019059023998984)]) workshops at PyData NYC this past week, and it got me to hacking on a PyMC3 version of the algorithm from [their recent paper](http://arxiv.org/abs/1804.06788) (also with Dan Simpson, Aki Vehtari, and Andrew Gelman).

The paper provides an algorithm, simulation based calibration (SBC), for checking whether an algorithm that produces samples from a posterior (like MCMC, ADVI, or INLA) might work for a given model. This calibration is independent of the observations for a model:

- we sample parameters `θ` from the prior, then use those samples to generate draws from the *prior predictive* distribution.
- For each draw `y` from the prior predictive distribution, we calculate, say, 100 draws from our posterior, using `y` as our observation.
- If you sort `θ` into the posterior samples, its position ("rank statistic") should be uniformly distributed.

Read the paper. It is very nice.

## Using the library

You can pip install and use the library right now!

```bash
pip install git+https://github.com/ColCarroll/simulation_based_calibration
```

See [the github project](https://github.com/ColCarroll/simulation_based_calibration) for sample usage.
