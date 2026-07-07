# ClusterFuzzLite build environment for the pure backend service functions.
# The OSS-Fuzz base-builder-python image ships Atheris, libFuzzer, and the
# `compile_python_fuzzer` helper (its own Python 3.11 — see build.sh for why we
# never `pip install .`). Digest-pinned (Scorecard Pinned-Dependencies); the
# image has no semver tag, so Dependabot (docker ecosystem, /.clusterfuzzlite)
# refreshes the digest.
FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:6768b69b56e3c6e654742e5109559fc37601b45d0cfe2e15ecb8e14f0e941a57

# The harnesses (backend/fuzz/*_fuzzer.py) import the pure modules under test
# (app.services.*) plus the real enums (app.models), so the whole backend tree
# is copied in. build.sh puts backend/ on PYTHONPATH and installs the hash-pinned
# import-deps from requirements.txt.
COPY backend $SRC/backend
COPY .clusterfuzzlite/build.sh $SRC/build.sh
COPY .clusterfuzzlite/requirements.txt $SRC/requirements.txt
WORKDIR $SRC/backend
