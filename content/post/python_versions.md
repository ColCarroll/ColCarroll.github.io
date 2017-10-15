+++
date = "2017-10-14T12:01:35-04:00"
draft = false
title = "Handling multiple python versions"
highlight = true
math = false
summary = "My setup for running multiple versions of Python"
tags = []
+++

My <a href="https://mediacloud.org/" target="blank_">new day job</a> is on python 3.5.2, 
<a href="https://github.com/pymc-devs/pymc3" target="blank_">PyMC3</a> runs on 2.7 and 3.6, and 
supports both `pip` and `conda` installations, and my 
<a href="https://github.com/ColCarroll?tab=repositories" target="blank_">personal projects</a>
are all over the place, but I think I have a great and simple set up for having sane and explicit
control of all the versions and dependencies.

The short version for setting all this up is to have bash functions for <b>n</b>ew 
<b>v</b>irtualenvs (`vn`), <b>a</b>ctivating <b>v</b>irtualenvs (`va`), <b>d</b>eactivating 
<b>v</b>irtualenvs (`vd`), and <b>dd</b>eleting <b>v</b>irtualenvs (`vdd`), along with 
<a href="https://github.com/pyenv/pyenv" target="blank_">`pyenv`</a> for handling the version of 
python I am using, and 
<a href="https://github.com/pyenv/pyenv-virtualenv" target="blank_">`pyenv-virtualenv`</a> because
`pyenv` does not play that well with `conda`.

## How to set it up!
```bash
$ brew install pyenv pyenv-virtualenv
```

Then add to `~/.bashrc`:
```bash
# virtualenv aliases
function virtualenv_name { echo "${PWD##*/}" ; }
function vn { pyenv virtualenv "$(virtualenv_name)" ; }
function va { pyenv activate "$(virtualenv_name)" ; }
alias vd="pyenv deactivate"
function vdd { pyenv uninstall "$(virtualenv_name)" ; }

# sets up pyenv and pyenv-virtualenv
export PYENV_ROOT=/usr/local/var/pyenv
if which pyenv > /dev/null; then eval "$(pyenv init -)"; fi
if which pyenv-virtualenv > /dev/null; then eval "$(pyenv virtualenv-init -)"; fi
```

This allows me to set up a new project with, for example

```bash
$ mkdir my_new_project
$ cd my_new_project
$ pyenv local 3.6.3
$ vn
Requirement already satisfied: setuptools in /usr/local/var/pyenv/versions/3.6.3/envs/my_new_project/lib/python3.6/site-packages
Requirement already satisfied: pip in /usr/local/var/pyenv/versions/3.6.3/envs/my_new_project/lib/python3.6/site-packages
$ va
$ (my_new_project) pip install whatever
```

If I had wanted a conda version, I could have used `pyenv local anaconda3-4.3.1` instead.

Note that your virtual environment is always the name of the directory you are in, and you cannot
have two different virtual environments in the same folder. I usually only develop `PyMC3` using
Python 3, and use travis and experience to make sure it works on Python 2.  If something is
funky though, I can delete one environment, swap to Python 2.7, and rebuild a fresh one to debug.

As an aside, pyenv does an incredible job of keeping up with the latest Python releases - the 
alpha candidate for Python 3.7.0 
<a href="https://github.com/pyenv/pyenv/commit/f9183b5f8ccaf2c0a5778dfb229c1cc85af85492" target="blank_">was up</a> a day after 
<a href="https://www.python.org/downloads/release/python-370a1/" target="blank_">it was released</a>.
