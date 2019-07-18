+++
title = "ArviZ: a unified library for Bayesian model criticism and visualization in Python"
date = 2018-10-05T12:00:40-04:00  # Schedule page publish date.
draft = false

# Talk start and end times.
#   End time can optionally be hidden by prefixing the line with `#`.
time_start = 2018-10-05T12:00:00-04:00
time_end = 2018-10-05T17:00:00-04:00

# Abstract and optional shortened version.
abstract = "Bayesian inference produces naturally high dimensional data: in the case of Markov chain Monte Carlo, it is common to produce multiple independent simulations (chains) to facilitate calculations like effective sample size and r-hat statistics. In this case, posterior samples are of dimension at least 2, and higher for multivariate random variables. Storing the data as an xarray dataset allows for labeled querying of this data, along with serialization, and attached metadata. ArviZ is a software library that stores these multiple datasets from inference using netCDF groups, which are themselves built with HDF5. The `InferenceData` class implements this functionality. By using netCDF, in addition to native handling of high dimensional data and being able to use existing serialization and deserialization function, all functions need be implemented only once. "
abstract_short = "Introducing the ArviZ software package for visualization, diagnostics, and criticism of Bayesian models."

# Name of event and optional event URL.
event = "PROBPROG 2018: The International Conference on Probabilistic Programming"
event_url = "https://probprog.cc"

# Location of event.
location = "MIT, Cambridge, MA"

# Is this a selected talk? (true/false)
selected = false

# Projects (optional).
#   Associate this talk with one or more of your projects.
#   Simply enter the filename (excluding '.md') of your project file in `content/project/`.
#   E.g. `projects = ["deep-learning"]` references `content/project/deep-learning.md`.
projects = []

# Tags (optional).
#   Set `tags = []` for no tags, or use the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags = []

# Links (optional).
url_pdf = "https://github.com/ColCarroll/probprog_poster"
url_slides = ""
url_video = ""
url_code = ""

# Does the content use math formatting?
math = false

# Does the content use source code highlighting?
highlight = true

# Featured image
# Place your image in the `static/img/` folder and reference its filename below, e.g. `image = "example.jpg"`.
[header]
image = ""
caption = ""

+++
