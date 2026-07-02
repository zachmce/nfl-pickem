# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities **privately** — do not open a public issue
or pull request for a suspected vulnerability.

Use GitHub's private vulnerability reporting for this repository:

1. Go to the **Security** tab of this repository.
2. Click **Report a vulnerability** (under "Advisories").
3. Provide a description, affected version/commit, and reproduction steps.

This opens a private advisory visible only to the maintainer. You'll receive an
acknowledgement, and we'll coordinate a fix and disclosure timeline with you.

If you're unable to use private advisories, you may open a minimal public issue
titled "Security contact request" (with **no** vulnerability details) asking the
maintainer to open a private channel.

## Supported Versions

This project is developed on a rolling basis; only the latest `main` is supported.
Fixes are applied to `main` and rolled into the next deployment.

## Scope

In scope: the application source in this repository (backend, frontend, bot) and
its CI/CD and container configuration.

Out of scope: vulnerabilities in third-party dependencies (report those upstream;
we track them via Dependabot, `pip-audit`, and Trivy), and issues requiring
physical access or a compromised maintainer account.

## What to Expect

- **Acknowledgement:** within a few days of a valid report.
- **Assessment & fix:** severity-dependent; we'll keep you updated in the advisory.
- **Credit:** we're happy to credit reporters in the advisory unless you prefer to
  remain anonymous.
