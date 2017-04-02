+++
date = "2017-04-02T15:33:57-04:00"
title = "Introducing Callisto"
tags = []
math = false
summary = "A command line utility to create kernels in Jupyter from virtual environments."

[header]

+++

# Callisto

Inspired by some personal frustrations around doing reproducible programming inside a [Jupyter](http://jupyter.org/) notebook, I built a small command line utility to easily create and delete kernels using virtual environments.  

Using this you can work on building a good, reusable project, manage dependencies with a virtualenv, but use that code in a Jupyter notebook.

The [readme](https://github.com/ColCarroll/callisto) gives a pretty good description of how to install.  Note the bit at the bottom about adding a directory to the `$PYTHONPATH` for that kernel.

Try it out with `pip install callisto`, activate a virtual environment, run
```python
$  callisto -p ~/path/to/project
```

Then happy hacking in your new Jupyter kernel.
