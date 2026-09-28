# ClusterFuzzLite build environment for the pure backend service functions.
# The OSS-Fuzz base-builder-python image ships Atheris, libFuzzer, and the
# `compile_python_fuzzer` helper (its own Python 3.11 — see build.sh for why we
# never `pip install .`). Digest-pinned (Scorecard Pinned-Dependencies); the
# image has no semver tag, so Dependabot (docker ecosystem, /.clusterfuzzlite)
# refreshes the digest.
FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:6a90ec89a18436e057a06f54266c3f120eb87d96c095b0c67a2b15cab33ec1f4

# The harnesses (backend/fuzz/*_fuzzer.py) import the pure modules under test
# (app.services.*) plus the real enums (app.models), so the whole backend tree
# is copied in. build.sh puts backend/ on PYTHONPATH and installs the hash-pinned
# import-deps from requirements.txt.
COPY backend $SRC/backend
COPY .clusterfuzzlite/build.sh $SRC/build.sh
COPY .clusterfuzzlite/requirements.txt $SRC/requirements.txt
WORKDIR $SRC/backend
