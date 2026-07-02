#!/bin/bash -eu
# ClusterFuzzLite build script — compile the Atheris harnesses.
#
# Deliberately does NOT `pip install .`: backend/pyproject.toml pins
# requires-python>=3.14, but the base-builder-python image ships Python 3.11, so
# an editable/project install is rejected by the requires-python gate. The pure
# service modules only need their import-time third-party deps — install just
# those (hash-pinned via requirements.txt, --require-hashes → Scorecard
# Pinned-Dependencies) and put backend/ on PYTHONPATH so `app.*`/`fuzz.*` import.
pip3 install --no-cache-dir --require-hashes -r "$SRC/requirements.txt"

export PYTHONPATH="$SRC/backend:${PYTHONPATH:-}"

# Compile each harness (backend/fuzz/*_fuzzer.py) into a standalone libFuzzer
# binary in $OUT. compile_python_fuzzer bundles via PyInstaller under the base
# image's Python, sidestepping the repo's 3.14 target.
for fuzzer in "$SRC"/backend/fuzz/*_fuzzer.py; do
  compile_python_fuzzer "$fuzzer"
done
