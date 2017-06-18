+++
date = "2017-06-18T22:16:15-04:00"
draft = false
title = "Releasing a project on PyPI"
highlight = true
math = false
summary = "The last mile in making a package pip installable."
+++

Caveat!  This is mostly me cargo-culting what worked for me.  The docs suggest using `twine`, which I couldn't get to work.

I'm assuming you've got a nice python package, a working `setup.py`, and are ready to release version 0.1 of `foo` on pypi:


0. Register at pypi.python.org and login
1. Run `python setup.py sdist`, to generate the `foo.egg-info/` directory.  Inside should be `PKG-INFO`.
2. Run `python setup.py bdist_wheel`, to generate `build/`, which can be ignored, and `dist/`.  Inside `dist/` are the python wheel (`foo-0.1-py2.py3-none-any.whl`) and the source distribution (`foo-0.1.tar.gz`)
3. Go to the "[Package submission](https://pypi.python.org/pypi?%3Aaction=submit_form)" (caution: link is annoying if you aren't logged in) and upload `PKG-INFO`.
4. This will create some pages for you on pypi, like https://pypi.python.org/pypi/foo/0.1, and direct you to a page where you can upload files.  Upload the wheel and source distribution and wheel from step 2, and you should be good to go.
5. Create a clean virtual environment and `pip install foo` to make sure nothing went wrong!

Note that if you make even a tiny change to anything, you'll need to upload a new release, so probably make sure you're happy before you start.
