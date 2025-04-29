# Contribution Guide

Thanks a lot for your interest and taking the time to contribute
to the Minterpy project!

This document provides short guidelines for contributing
to the Minterpy project.
For a more comprehensive guide, please refer to the online documentation
([stable](https://minterpy-project.github.io/minterpy/stable/contributors/index.html)
or [latest](https://minterpy-project.github.io/minterpy/latest/contributors/index.html))

## Installation

This installation guide is focused on development.
For installing Minterpy in production environment, check out [README.md](./README.md).

### Obtaining the source

To obtain the latest source,
clone the Minterpy repository from [GitHub](https://github.com/minterpy-project/minterpy):

```bash
git clone https://github.com/minterpy-project/minterpy
```

By default, the cloned branch is the `dev` branch (i.e., the latest development
version).

We recommend always pulling the latest commit:

```bash
git pull origin dev
```

You are not allowed to directly push to `dev` or `main` branch.
Please follow the instructions under [Branching workflow](#branching-workflow).

### Virtual environments

Following a best practice in Python development,
we strongly encourage you to create and use virtual environments for development and production runs.
A virtual environment encapsulates the package and all dependencies without messing up your other Python installations.

The following instructions should be executed from the Minterpy source directory.

#### Using [venv](https://docs.python.org/3/tutorial/venv.html) from the python standard library:

1. Build a virtual environment:

    ```bash
    python -m venv <your_venv_name>
    ```

   Replace `<your_venv_name>` with an environment name of your choice.

2. Activate the environment you just created:

    ```bash
    source <your_venv_name>/bin/activate
    ```

    Replace <your_venv_name> with your desired environment name.

3. To deactivate the virtual environment, type:

    ```bash
    deactivate
    ```

#### Using [virtualenv](https://virtualenv.pypa.io/en/latest/):

1. Building a virtual environment:

    ```bash
    virtualenv <your_venv_name>
    ```

   Replace `<your_venv_name>` with an environment name of your choice.

2. Activate the environment you just created:

    ```bash
    source <your_venv_name>/bin/activate
    ```

    Replace <your_venv_name> with your desired environment name.

3. To deactivate the virtual environment, type:

    ```bash
    deactivate
    ```

#### Using [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv):

1. Building the virtual environment:

    ```bash
    pyenv virtualenv 3.8 <your_venv_name>
    ```

   Replace `<your_venv_name>` with an environment name of your choice.

2. Activate the environment you just created:

    ```bash
    pyenv local <your_venv_name>
    ```

    Replace <your_venv_name> with your desired environment name.

    This command creates a hidden `.python_version` file containing
    a "link" to the actual virtual environment managed by `pyenv`.

3. To "deactivate" the virtual environment just remove this hidden file:

    ```bash
    rm .python_version
    ```

#### Using [conda](https://conda.io/projects/conda/en/latest/index.html):

1. Create an environment `minterpy` with the help of [conda]https://conda.io/projects/conda/en/latest/index.html)
   and the file `environment.yaml`
   (included in the source distribution of `minterpy`):

   ```bash
   conda env create -f environment.yaml
   ```

   The command creates a new conda environment called `minterpy`.

2. Activate the new environment with:

   ```bash
   conda activate minterpy
   ```

   You may need to initialize conda env;
   follow the instructions printed out or read the conda docs.

3. To deactivate the conda environment, type:

    ```bash
    conda deactivate
    ```

### Installation

We recommend using [pip](https://pip.pypa.io/en/stable/) from within a virtual environment (see above)
to install Minterpy.

To install Minterpy from source, type:

```bash
pip install [-e] .[all,dev,docs]
```

where the flag `-e` means the package is directly linked into the Python site-packages.
The options `[all,dev,docs]` refer to the requirements defined in the `options.extras_require` section in `setup.cfg`.

**Note**: **Do not** use `python setup.py install`,
since the file `setup.py` will not be present for every build of the package.

### Troubleshooting: pytest with venv (*not* conda)

After installation, you might need to restart your virtual environment
since the `pytest` command uses the `PYTHONPATH` environment variable which not automatically change to your virtual environment.

To restart your virtual environment created by `venv`, type:

```bash
deactivate && source <your_venv_name>/bin/activate
```

or run `hash -r` instead.

This issue does not seem to occur for environments created by Conda.

### Dependency management & reproducibility (conda)

Here are some recommendations for managing dependency
and maintaining reproducibility of your Minterpy development environment:

1. Always keep your abstract (unpinned) dependencies updated in `environment.yaml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.

2. Create concrete dependencies as `environment.lock.yaml`
   for the exact reproduction of your environment with:

   ```bash
   conda env export -n minterpy -f environment.lock.yaml
   ```

   For multi-OS development, consider using `--no-builds` during the export.

3. Update your current environment with respect to a new `environment.lock.yaml` using:

   ```bash
   conda env update -f environment.lock.yaml --prune
   ```

## Testing

We use [pytest](https://docs.pytest.org/en/6.2.x/) to run the unit tests of Minterpy.
The unit tests themselves must always be placed into the `tests` directory.
To run all tests, type:

```bash
pytest
```

from within the Minterpy source directory.

If you want to run the tests of a particular module,
for instance the `multi_index_utils.py` module, execute:

```bash
pytest tests/test_multi_index_utils.py
```

When running `pytest`, the coverage tests are automotically performed.
A summary of the coverage test is printed out in the terminal.
Furthermore, you can find an HTML version of the coverage test results
in `htmlcov/index.html`.

### Writing new tests

We strongly encourage you to use the capabilities of `pytest` for writing the unit tests

Be aware of the following points:

- the developer of the code should write the tests
- test the behavior you expect from your code, not breaking points
- use as small samples as possible
- unit tests do *not* test if the code works, they test if the code *still* works
- the coverage should always be as high as possible
- BUT, even 100% coverage does not mean, there is nothing missed (buzz: edge case!)

For additional reference on how to write tests, have a look at the following resources:

- [Pytest: Examples and customization tricks](https://docs.pytest.org/en/6.2.x/example/index.html)
- [Effective Python Testing with Pytest](https://realpython.com/pytest-python-testing/)
- [Testing best practices for ML libraries](https://towardsdatascience.com/testing-best-practices-for-machine-learning-libraries-41b7d0362c95)

## Documentation

This section provides some information about building and contributing
to the documentation.

### Install dependencies

Building the documentation requires additional dependencies.
You can install Minterpy from source with all the dependencies for building
the documentation as follows:

```bash
pip install .[docs]
```

### Building the documentation

We use [sphinx](https://www.sphinx-doc.org/en/master/) to build the `minterpy` docs.
To build the docs in HTML format, run the following command:

```bash
sphinx-build -M html docs docs/build
```

Alternatively, you can build the documentation using the supplied Makefile.
For that, you need to navigate to the `docs` directory and run the `make` command in Linux/mac OS or `make.bat` in Windows:

```bash
cd docs
make html
```

The command builds the docs and stores it in in `docs/build`.
You may open the docs using a web browser of your choice by opening `docs/build/html/index.html`.

You can also generate the docs in PDF format using `pdflatex` (requires a LaTeX distribution installed in your system):

```bash
cd docs
make latexpdf
```

The command builds the docs as a PDF document and stores it along with all the LaTeX source files in `docs/build/latex`.

### Documentation source organization

The source files for the documentation are stored in the `docs` directory.
The Sphinx configuration file is `docs/conf.py`,
and the main index file of the docs is `docs/index.rst`.

You can find more information about the Minterpy documentation
in the Contributors Guide (
[stable](https://minterpy-project.github.io/minterpy/stable/contributors/contrib-docs/index.html)
or [latest](https://minterpy-project.github.io/minterpy/latest/contributors/contrib-docs/index.html)).

## Code style

To ensure the readability of the codebase,
we are following a common code style for Minterpy.
Our long-term goal is to fulfill the [PEP8](https://www.python.org/dev/peps/pep-0008/) regulations.
For the build system, it is recommended to follow [PEP517](https://www.python.org/dev/peps/pep-0517/)
and [PEP518](https://www.python.org/dev/peps/pep-0518/).
However, since these requirements are very challenging,
we use [black](https://github.com/psf/black) to enforce the code style of Minterpy.

During the development process,
you can check the format using [pre-commit](https://pre-commit.com) (see below) and

In the development process, one can check the format using
and the hooks defined in `.pre-commit-config.yaml`.
For instance running `black` for the whole `minterpy` code, just run

```bash
pre-commit run black --all-files
```

For now, it is recommended to run single hooks.

## Pre-commit

For on-going developments, it is recommended to run all pre-commit-hooks
every time before committing some changes to your branch.

Install the pre-commit hooks by running:

```bash
pre-commit install
```

If you want to disable the pre-commit script, type:

```bash
pre-commit uninstall
```

To run all hooks defined in `.pre-commit-config.yaml`, type:

```bash
pre-commit run --all-files # DON'T DO THIS IF YOU DON'T KNOW WHAT HAPPENS
```

In the current state of the code, you should use this with caution
since it might change code in the manner that it breaks (see below).

Down the road, we shall try to fulfill the full set of pre-commit hooks.
However, further developments shall try to fulfil the full set of pre-commit-hooks.

### Currently defined hooks

The following hooks are defined:

- [black](https://github.com/psf/black): a straightforward code formatter;
  it modifies the code in order to fulfill the format requirement.
- [pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks): A collection of widely used hooks;
  see their repository for more informations.
- [isort](https://github.com/PyCQA/isort): sorts the import statements;
  changes the code (**NOTE**: **Do not** run and commit the changes; it may break the current version.)
- [pyupgrade](https://github.com/asottile/pyupgrade): convert the syntax from Python2 to Python3.
  It's nice if you use code from an old post in stackoverflow ;-)
- [setup-cfg-fmt](https://github.com/asottile/setup-cfg-fmt): formats the `setup.cfg` file for consistency.
- [flake8](https://github.com/pycqa/flake8): a collection of hooks to ensure most of the PEP8 guidelines are satisfied.
  The concrete checks are defined in the `setup.cfg[flake8]`.
- [mypy](https://github.com/python/mypy): a static type checker;
  `mypy` itself is configured in the `setup.cfg[mypy-*]`.
- [check-manifest](https://github.com/mgedmin/check-manifest):
  checks if the `MANIFEST.in` is in a proper state.
  This ensures proper builds for uploading the package to [PyPI](https://pypi.org).
  This is configured in `setup.cfg[check-manifest]`.

In case you're using pre-commit hooks, be sure to run the test again before
committing or pushing any changes.

## Code development

### Version control

We only use [git](https://git-scm.com/) to version control Minterpy.
The main repository for development is
on [GitHub](https://github.com/minterpy-project/minterpy).
Moreover, the releases and the development branch
are also mirrored into the [CASUS GitHub](https://github.com/casus/minterpy) repository.

The latest release of Minterpy is available in [PyPI](https://pypi.org/project/minterpy/).

### Branching workflow

We follow the structure of [Gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) for our branching workflow.
There are three types of branches in this workflow:

1. `main`  branch:
    On this branch, only the releases are stored.
    This means, on this branch, one has only fully tested, documented and cleaned up code.
2. `dev` branch:
    On this branch, the development version are stored.
    At any given time, the branch must pass all the tests.
    This also means that on this branch, there is always a running version of `minterpy`
    even if the code and the docs are not in a "release state."
3. `feature` branches:
    On these branches, all the features and code developments happen.
    `feature` branches must be created from the `dev` branch (not from `main`).

Based on this workflow, you can freely push, change, and merge *only* on the `feature` branches.
Furthermore, your feature branch is open to every developers in the `minterpy` project.

Once the implementation of a feature is finished,
you can merge the `feature` branch to the `dev` branch via a pull request.
The project maintainers will merge your pull request once the request is reviewed.
In general, you cannot merge your `feature` branch directly to the `dev` branch.

Furthermore, as a contributor, you cannot merge directly to the `main` branch
and you cannot make a pull request for that.
Only the project maintainers can merge the `dev` to the `main` branch
following the release procedure of [Gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).

We manage the bug fixes on every branch separately with the relevant developers,
usually via `hotfix` branches to implement the patches.

More details can be found in the online documentation
([stable](https://minterpy-project.github.io/minterpy/stable/contributors/development-environment.html#about-the-branching-model)
or [latest](https://minterpy-project.github.io/minterpy/latest/contributors/development-environment.html#about-the-branching-model))


## Source organization

```
├── .gitignore              <- ignored files/directories if `git add/commit`
├── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Contribution guidelines (this file).
├── environment.yaml        <- The conda environment file for reproducibility.
├── LICENSE                 <- License as chosen on the command-line.
├── MANIFEST.in             <- Keep track of (minimal) source distribution files
├── pyproject.toml          <- Specification build requirements
├── README.md               <- The top-level README for developers.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `python setup.py develop` to install for development or
|                              or create a distribution with `python setup.py bdist_wheel`.
├── .github                 <- scripts for GitHub actions.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── src
│   └── minterpy            <- Actual Python package where the main functionality goes.
└── tests                   <- Unit tests which can be run with `pytest`.
```
