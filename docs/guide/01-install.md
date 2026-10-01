# 1. Install

chairlift needs Python 3.12 or later and [uv](https://docs.astral.sh/uv/). There are three ways to install it,
depending on what you are doing.

## Working on chairlift itself

```bash
git clone git@github.com:lanteignel93/chairlift.git
cd chairlift
uv sync                      # .venv from uv.lock, dev tools included
uv run chairlift --version   # chairlift, version 0.0.1
uv run pytest                # the whole suite, walkthroughs included; about ten seconds
uv run prek install          # once per clone: ruff, basedpyright and typos on commit, pytest on push
```

If uv warns `Failed to hardlink files`, the uv cache is on a different filesystem from the checkout. Set
`export UV_LINK_MODE=copy` in your shell profile.

## Using chairlift from a study repository (the normal case)

A study lives in its own repository and depends on chairlift. Run `chairlift` from the study's environment, so that
the study's own package and its dependencies are importable by the factory:

```bash
cd my-study
uv add "chairlift @ git+ssh://git@github.com/lanteignel93/chairlift"   # pinned in the study's uv.lock
# or, while developing both side by side:
uv add --editable ../chairlift
uv run chairlift run my_study.study:pipeline
```

The study's `uv.lock` then pins the exact chairlift commit. Every run records the code identity
(`chairlift <version> @ <commit>[+dirty]`) and an environment hash ([5](05-reproducibility.md)), so a record always
says which chairlift built it.

## As a command on your PATH

```bash
uv tool install --editable /path/to/chairlift
chairlift config show
```

This is convenient for `watch`, `status`, `runs`, `log` and `config`, which read files and import no study. Running a
study this way works only for `path/to/file.py:factory` references whose imports are already installed in the tool's
environment, so prefer `uv run chairlift` inside the study repository for `run`, `plan` and `sweep`.

## Check the install

```bash
uv run chairlift config check        # ok · 0 file(s) · profile -
uv run chairlift run chairlift.verify.toy:pipeline --root /tmp/chairlift-check
```

The second command builds the toy study in a throwaway directory: seven stages, all `ran`. Run it again and all
seven are `hit`. Next: [configure the machine](02-configure.md).
