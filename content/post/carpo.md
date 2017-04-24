+++
title = "Introducing Carpo"
tags = []
highlight = true
math = false
date = "2017-04-23T21:47:13-04:00"
summary = "A command line utility to run, profile, and save Jupyter notebooks."

[header]

+++


# Carpo

Continuing the tour of Jupiter's moons, I factored out a small utility script from [pymc3](https://pymc-devs.github.io/pymc3/) that I was using to make sure Jupyter](http://jupyter.org/) notebooks kept up with the rapidly changing API, and made it into an easy CLI.  The CLI looks up git information if available, and writes down the outcome from each run into a sqlite database.  This lets the user start and stop runs without losing progress, and do rough benchmarking between git shas.

I'm honestly not sure the use case beyond `PyMC3`'s, where there are ~50 example notebooks, some of which take tens of minutes to run, but it is on `pypi` for general consumption:  `pip install carpo` will activate it, and the only commands right now are:

```python
$  carpo run ~/path/to/notebooks/*.ipynb
...
$  carpo show ~/path/to/notebooks/*.ipynb
...
```
Also use `carpo --help` to see details.
