#!/usr/bin/env bash
# Audits the locked backend + frontend deps and bumps every package that has a
# fixed version, ignoring the Dependabot cooldown: deps-audit and osv-scan fail
# CI on ANY advisory, so a fixable one blocks every PR until it is bumped.
# Writes $OUT/report.md; appends findings=/changed= to $GITHUB_OUTPUT if set.
# Run from the repo root. Called by .github/workflows/security-audit.yml.
set -euo pipefail
mkdir -p "${OUT:?set OUT to a scratch directory}"
out=$(cd "$OUT" && pwd)

pip_audit() {
  (cd backend && uv export --format requirements-txt --no-emit-project --no-hashes -q -o "$out/req.txt")
  rm -f "$1"
  # pip-audit exits 1 on findings; a missing JSON is the real error.
  (cd backend && uv run pip-audit -r "$out/req.txt" -f json -o "$1" >/dev/null 2>&1) || true
  [ -s "$1" ] || { echo "::error::pip-audit produced no report"; exit 1; }
}

npm_audit() {
  (cd frontend && npm audit --package-lock-only --json >"$1" 2>/dev/null) || true
  jq -e '.metadata.vulnerabilities.total' "$1" >/dev/null || { echo "::error::npm audit produced no report"; exit 1; }
}

py_table() {
  jq -r '.dependencies[] | select(.vulns | length > 0) | .name as $n | .version as $v
    | .vulns[] | "| \($n) | \($v) | \(([.id] + .aliases) | join(", ")) | \(.fix_versions | join(", ") | if . == "" then "none" else . end) |"' "$1"
}

npm_table() {
  jq -r '.vulnerabilities[] | "| \(.name) | \(.range) | \(.severity) | \(if .fixAvailable == false then "none" else "yes" end) |"' "$1"
}

pip_audit "$out/py-before.json"
npm_audit "$out/npm-before.json"

# Smallest version that fixes every advisory on the package; None = no fix released.
specs=$(cd backend && uv run python - "$out/py-before.json" <<'PY'
import json, sys
from packaging.version import Version

for dep in json.load(open(sys.argv[1]))["dependencies"]:
    cur, need = Version(dep["version"]), None
    for vuln in dep.get("vulns", []):
        fixes = sorted(v for v in map(Version, vuln["fix_versions"]) if v > cur)
        if not fixes:
            need = None
            break
        need = max(need or fixes[0], fixes[0])
    if need:
        print(f"{dep['name']}=={need}")
PY
)

lock_failed=""
for spec in $specs; do
  (cd backend && uv lock -q --upgrade-package "$spec") || lock_failed+="$spec "
done

if [ "$(jq '.metadata.vulnerabilities.total' "$out/npm-before.json")" -gt 0 ]; then
  (cd frontend && npm audit fix --package-lock-only --ignore-scripts >/dev/null 2>&1) || true
fi

pip_audit "$out/py-after.json"
npm_audit "$out/npm-after.json"

py_before=$(py_table "$out/py-before.json")
npm_before=$(npm_table "$out/npm-before.json")
residual="$(py_table "$out/py-after.json")$(npm_table "$out/npm-after.json")"
findings=false; [ -n "$py_before$npm_before" ] && findings=true
changed=true; git diff --quiet -- backend/uv.lock frontend/package.json frontend/package-lock.json && changed=false

{
  echo "Audit of \`main\` at ${GITHUB_SHA:-$(git rev-parse HEAD)}."
  if [ -n "$py_before" ]; then
    printf '\n### Backend (pip-audit; CI fails on any advisory)\n\n| Package | Locked | Advisory | Fixed in |\n|---|---|---|---|\n%s\n' "$py_before"
  fi
  if [ -n "$npm_before" ]; then
    printf '\n### Frontend (npm audit; osv-scan fails on any advisory)\n\n| Package | Range | Severity | Fix |\n|---|---|---|---|\n%s\n' "$npm_before"
  fi
  [ -n "$lock_failed" ] && printf '\n`uv lock` could not apply: `%s`\n' "$lock_failed"
  if [ -n "$residual" ]; then
    printf '\n### Still vulnerable after the auto-fix\n\nNo fix applied for these. Bump by hand, or add a rationale `--ignore-vuln` in ci.yml if no fix exists.\n\n| Package | Version | Advisory / severity | Fix |\n|---|---|---|---|\n%s\n' "$residual"
  fi
} >"$out/report.md"

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "findings=$findings" >>"$GITHUB_OUTPUT"
  echo "changed=$changed" >>"$GITHUB_OUTPUT"
fi
echo "findings=$findings changed=$changed"
