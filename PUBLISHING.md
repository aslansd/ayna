# Publishing a release

## Steps

1. **Bump the version** in `src/ayna/__init__.py` (`__version__`). It is the
   only place: `pyproject.toml` reads it dynamically.
2. **Write the changelog entry** in `CHANGELOG.md` before doing anything else.
   If it is hard to write, the release is not ready.
3. **Run the tests**:
   ```bash
   pytest -q                                  # core suite, must be all green
   # in the CtD environment (see TESTING.md):
   AYNA_CTD_WALKTHROUGH=/path/to/walkthrough_models pytest tests/test_ctd_live.py -v
   ```
4. **Re-run the validation numbers quoted in the README and VALIDATION.md**
   if any detector or threshold changed:
   ```bash
   ayna validate --n 10 --seed 7               # synthetic test corpus
   # and the held-out sweeps: experiments/ctd_tasks/README.md
   ayna ctd $W/node_sweep --model-type SAE --tt $W/tt_3bff --out reports/node
   ayna ctd $W/inputinf_sweep --model-type LFADS --tt $W/tt_3bff --out reports/lfads
   ```
   Paste the new tables. Do not leave old numbers next to new code.
5. **Build and check**:
   ```bash
   rm -rf dist build src/*.egg-info
   python -m build
   twine check dist/*
   ```
6. **Install the wheel in a fresh virtualenv** and run the README quickstart
   there. Source-tree tests cannot catch packaging mistakes such as a missing
   module or a wrong entry point.
   ```bash
   python -m venv /tmp/ayna-check && /tmp/ayna-check/bin/pip install dist/ayna-*.whl
   cd /tmp && /tmp/ayna-check/bin/ayna synth field --faults feature_invention misattribution=0.8 \
       && /tmp/ayna-check/bin/ayna scan datasets/field.npz
   ```
7. **Upload, tag, push**:
   ```bash
   twine upload dist/*
   git tag -a v0.3.0 -m "ayna 0.3.0"
   git push && git push --tags
   ```
8. **Verify from PyPI**. The index can serve the old version for a few minutes:
   ```bash
   pip install --no-cache-dir -U ayna && ayna --version
   ```

## Cautions

- A version number can never be reused on PyPI, even after deleting the
  release. If a release is broken, ship the next patch version.
- Use a project-scoped API token for `ayna`, not an account-wide one. If you
  keep it in `~/.pypirc`, make that file readable only by you (`chmod 600`).
- TestPyPI needs a separate account and token. For a small pure-Python package,
  the fresh-virtualenv check in step 6 catches the same problems.

## Licensing

`LICENSE` holds the full Apache-2.0 text, and `pyproject.toml` declares it as
an SPDX expression (`license = "Apache-2.0"`). ayna ports metric definitions
from the CtD paper and code (BSD-3-Clause) and cites them in the README. No
CtD code or data ships with ayna. The CtD models are downloaded by the user.
