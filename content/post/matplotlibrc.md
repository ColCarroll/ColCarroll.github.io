+++
date = "2017-05-20T12:01:35-04:00"
draft = false
title = "Setting the matplotlib backend"
highlight = true
math = false
summary = "`python -c 'import matplotlib; print(matplotlib.matplotlib_fname())'`"
tags = []
+++

Running matplotlib on OS X, occasionally there will be a cryptic error about installing as a framework.  See [here](https://matplotlib.org/faq/osx_framework.html) for a more precise discussion.  The fix I typically use, and which is hard to find (on [this](http://matplotlib.org/users/customizing.html) page) is to run
```python
import matplotlib
matplotlib.matplotlib_fname()
```
then go to the printed file, find the line that says `'macosx'`, and change it to `'Agg'`.


Or, as a bash command liner, `python -c 'import matplotlib; print(matplotlib.matplotlib_fname())'`
