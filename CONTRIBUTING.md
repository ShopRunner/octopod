# Contributing and Making PRs

## How to Contribute

We welcome contributions in the form of issues or pull requests! 

We want this to be a place where all are welcome to discuss and contribute, so please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms. Find the Code of Conduct in the ``CONDUCT.md`` file on GitHub or in the Code of Conduct section of read the docs.

If you have a problem using Octopod or see a possible improvement, open an issue in the GitHub issue tracker. Please be as specific as you can.

If you see an open issue you'd like to be fixed, take a stab at it and open a PR!


Pull Requests
------------------
To create a PR against this library, please fork the project and work from there.

Steps
++++++

1. Fork the project via the Fork button on Github

2. Clone the repo to your local disk.

3. Create a new branch for your PR.

```
    git checkout -b my-awesome-new-feature
```
4. Install requirements (probably in a virtual environment)

```
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements-dev.txt
    pip install -e .
```
5. Develop your feature
   
6. Submit a PR to master! Someone will review your code and merge your code into master when it is approved. 
   
PR Checklist
+++++++++++++

- Ensure your code has followed the Style Guidelines below
- Run the linter on your code

```
    source venv/bin/activate
    flake8 octopod tests
```
- Make sure you have written unittests where appropriate
- Make sure the unittests pass

```
    source venv/bin/activate
    pytest -v
```
- Update the docs where appropriate. You can rebuild them with the commands below.

```
     cd docs/
     make html
     open build/html/index.html
```
- Update the CHANGELOG

Style Guidelines
++++++++++++++++++++++++++

For the most part, this library follows PEP8 with a couple of exceptions. 

- Indent with 4 spaces
- Lines can be 100 characters long
- Docstrings should be [numpy style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html) docstrings.
- Your code should be Python 3 compatible
- When in doubt, follow the style of the existing code
- We prefer single quotes for one-line strings unless using double quotes allows us to avoid escaping internal single quotes.
