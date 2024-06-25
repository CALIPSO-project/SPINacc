# Contributing to the `SPINacc` package

### Reporting bugs

If you find a bug in SPINacc, please report it on the [GitHub issue
tracker](https://github.com/CALIPSO-project/SPINacc/labels/bug).


### Suggesting enhancements

If you want to suggest a new feature or an improvement of a current
feature, you can submit this on the [issue
tracker](https://github.com/CALIPSO-project/SPINacc/issues).


### Submitting a pull request

To contribute code to SPINacc, create a pull request. If you want to
contribute, but are unsure where to start, have a look at the [issues
labelled "good first
issue"](https://github.com/CALIPSO-project/SPINacc/labels/good%20first%20issue).
For substantial changes/contributions, please start with an issue.

> On opening a pull request, unit tests will run on GitHub CI. You can
> click on these in the pull request to see where (if anywhere) the tests
> are failing.

## Quick Start

 Getting a simple development environment should involve:

* `SPINacc` has been tested on `python==3.9.*`.

```sh
# Create a virtual environment
python3 -m venv ./venv3
source venv/bin/activate
# Download the repository
git clone git@github.com:CALIPSO-project/SPINacc.git
# Move into the repository root
cd SPINacc
# Install the package requirements
pip install -r requirements.txt
```

### Code quality checks

The package includes configuration for a set of `pre-commit` hooks to ensure code
commits meet common community quality standards. The `pre-commit` tool blocks `git`
commits until all hooks pass. You should install these to ensure that your commited code
meets these standards. To install, add the following to your virtual environment:

```sh
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
