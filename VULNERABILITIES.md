# 🔐 Security Vulnerability Report — MedAdhere AI

**Audit date:** 2026-06 · **Scope:** entire repository (`app/`, `data/`, `models/`, `notebooks/`, dependency set)
**Tooling:** `pip-audit` (OSV/PyPI advisory DB), `bandit` (static analysis, severity ≥ LOW), manual code review
**Tracking:** issues are tracked Linear-style below (IDs `MED-SEC-n`, priority, status). This file is the
single source of truth for security debt in the repo.

---

## Issue Board (Linear-style tracking)

| ID | Title | Priority | Status | Labels |
|---|---|---|---|---|
| MED-SEC-1 | Unsafe deserialization: joblib pickle artifacts loaded without integrity verification | 🔴 High | 🟡 Mitigated (partial) | `security` `deserialization` `supply-chain` |
| MED-SEC-2 | 70 known CVEs in the dependency tree (10 packages, unconstrained floors) | 🔴 High | ✅ Done | `security` `dependencies` `supply-chain` |
| MED-SEC-3 | Information disclosure: full stack traces shown to end users on prediction failure | 🟠 Medium | ✅ Done | `security` `info-disclosure` `ux` |
| MED-SEC-4 | HTML injection surface: dynamic values interpolated into `unsafe_allow_html` blocks unescaped | 🟠 Medium | ✅ Done | `security` `xss` `defense-in-depth` |
| MED-SEC-5 | Static analysis (bandit) clean — no actionable findings | ⚪ Info | ✅ Done | `security` `audit` |

**Legend:** Priority 🔴 High = exploitable / high impact · 🟠 Medium = defense-in-depth or limited exploitability · ⚪ Info = observation

**Status legend:** ✅ Done = fully fixed · 🟡 Mitigated (partial) = exploitability reduced, residual risk documented in the issue body

---

## MED-SEC-1 — Unsafe deserialization of model artifacts

- **Severity:** High
- **Location:** `app/app.py` → `load_ml_pipeline()` (`joblib.load(MODEL_PATH)`)
- **Description:** `medication_adherence_model.pkl` and `preprocessing_pipeline.pkl` are Python
  pickle files. **Deserializing a pickle executes arbitrary code** — anyone who can modify the
  artifact (supply-chain compromise, malicious contributor, tampered deployment bundle) achieves
  remote code execution on the host running the Streamlit app. The app loaded the pickle with no
  integrity check whatsoever.
- **Fix (applied — mitigates but does not fully close the issue):**
  - `app/integrity.py` verifies every artifact before `joblib.load()`; `app/app.py` refuses the
    load on any failure.
  - SHA-256 checksums of both artifacts are committed in `models/checksums.json` (schema 2),
    reliably catching *accidental* corruption and partial uploads.
  - **HMAC tier for deployments:** when `MEDADHERE_MODEL_KEY` is set in the environment, the app
    additionally verifies a per-artifact HMAC-SHA256 keyed with that secret (constant-time
    comparison, fail-closed if the manifest lacks HMAC entries). The key lives only in the
    deployment environment, so an attacker who swaps **both the artifact and the manifest** still
    cannot forge the HMAC. `models/update_checksums.py` emits HMAC entries when run with the key
    set; `notebooks/build_pipeline.py` calls it automatically after every retrain.
- **Residual risk (why status is Mitigated, not Done):**
  - Without `MEDADHERE_MODEL_KEY` configured, verification degrades to the in-repo checksum,
    which is not an independent trust anchor — an attacker who can replace a `.pkl` can also
    update `checksums.json`, and such a commit passes verification.
  - Verification runs inside the same deployable unit as the app: an attacker who controls the
    whole repository/deployment can also modify the verification code itself. Complete closure
    requires eliminating executable deserialization (skops/ONNX) and/or verifying signatures
    in the deployment pipeline, outside the application.
  - Both follow-ups are tracked in [Appendix B](#appendix-b--remediation-checklist-for-maintainers).

## MED-SEC-2 — 70 known vulnerabilities in the dependency tree

- **Severity:** High (aggregate)
- **Location:** `requirements.txt` (open-ended `>=` floors)
- **Evidence (`pip-audit -r requirements.txt`):** *Found 70 known vulnerabilities in 10 packages*
  — streamlit (2), pillow (52), requests (1), urllib3 (5), notebook (1), jupyter-server (2),
  anyio (2), bleach (2), pyarrow (1), soupsieve (2). Full ID/version/fix-version table reproduced
  in [Appendix A](#appendix-a--raw-pip-audit-output).
- **Root cause:** every requirement used an open `>=` floor, so any resolver — including one
  resolving years later — will happily install a version with publicly known CVEs.
- **Fix (applied):**
  - `requirements.txt` now carries fixed-version security floors for all 10 affected packages
    (e.g. `streamlit>=1.66.1`, `pillow>=13.0.0`, `urllib3>=2.8.0`), so the default install is
    secure with no extra flags. On older Python versions resolution fails closed (no vulnerable
    install) rather than silently proceeding.
  - Floor compatibility was verified against real PyPI metadata: streamlit 1.54–1.60 cap
    `pillow<13`, so streamlit was bumped to `>=1.66.1` (latest, the first line that permits
    `pillow 13`); streamlit 1.66.1's own dependency set (pillow<14, pyarrow<27, requests<3,
    anyio<5, pandas<4) is compatible with every pin. Streamlit 1.66.1 and pillow 13 require
    **Python ≥3.11**, so the project floor moved from 3.10 to 3.11 (README updated).
  - `requirements.txt` documents the re-audit command (`pip-audit -r requirements.txt`, to be
    run on Python ≥3.11).
- **Environment caveat:** the audit sandbox exposes only Python 3.9 and a package mirror that
  predates the fixed releases (its newest pillow is 11.3.0, newest streamlit 1.50.0), so the
  pinned set could not be installed and smoke-tested locally; co-installability was verified
  against real PyPI metadata instead (every pinned package exists and the floors are mutually
  compatible with streamlit 1.66.1's dependency set). Run `pip install -r requirements.txt` and
  `pip-audit -r requirements.txt` on Python ≥3.11 with full PyPI access to confirm.

## MED-SEC-3 — Stack traces exposed to end users

- **Severity:** Medium
- **Location:** `app/app.py` prediction handler (`st.exception(e)` and
  `st.error(f"Prediction Pipeline Error: {e}")`)
- **Description:** On any prediction failure the app rendered the raw exception and full Python
  traceback in the browser, leaking internal paths, library internals, and pipeline structure —
  useful reconnaissance for an attacker and confusing/panic-inducing for a clinical user.
- **Fix (applied):** full traceback is now logged server-side via the `logging` module; the user
  sees only a generic, actionable message. Artifact-missing/integrity errors likewise show a
  generic message with details in server logs (no absolute paths in the UI).

## MED-SEC-4 — Unescaped dynamic values in HTML blocks

- **Severity:** Medium (defense-in-depth)
- **Location:** `app/components.py` → `render_header()`, `render_insights()`;
  `app/app.py` → model-specification card (`metadata['model_name']`)
- **Description:** Dynamic values (patient profile fields, model metadata) were interpolated into
  `st.markdown(..., unsafe_allow_html=True)` f-strings without escaping. Current inputs arrive
  only from fixed widget sets, but `st.session_state` is mutable and any future change (upload
  forms, query-param presets, REST input) turns this straight into stored XSS in a
  healthcare-context app.
- **Fix (applied):** all dynamic interpolations are now passed through `html.escape()`
  (`_esc()` helper in `components.py`); static, developer-authored markup is unchanged.

## MED-SEC-5 — Static analysis observations

- **Severity:** Info
- **Result:** `bandit -r app data notebooks` reports **zero findings** (no hardcoded secrets, no
  `eval`/`exec`, no `shell=True` subprocesses, no weak hashing). The dataset generator uses only
  synthetic data (no PII), which keeps the project out of HIPAA/GDPR processing scope — the
  README's disclosure on this point is accurate.

---

## Appendix A — Raw pip-audit output

```
Name           Version ID                  Fix Versions
-------------- ------- ------------------- ------------
streamlit      1.50.0  PYSEC-2026-212      1.53.1
streamlit      1.50.0  PYSEC-2026-2285     1.54.0
pillow         11.3.0  PYSEC-2026-165      12.2.0
pillow         11.3.0  PYSEC-2026-2250     12.2.0
pillow         11.3.0  PYSEC-2026-2253     12.3.0
pillow         11.3.0  PYSEC-2026-2251     12.2.0
pillow         11.3.0  PYSEC-2026-2255     12.3.0
pillow         11.3.0  PYSEC-2026-2257     12.3.0
pillow         11.3.0  PYSEC-2026-2256     12.3.0
pillow         11.3.0  PYSEC-2026-2254     12.3.0
pillow         11.3.0  PYSEC-2026-2252     12.2.0
pillow         11.3.0  PYSEC-2026-2249     12.1.1
pillow         11.3.0  PYSEC-2026-2874     12.2.0
pillow         11.3.0  PYSEC-2026-3453     12.3.0
pillow         11.3.0  PYSEC-2026-3451     12.3.0
pillow         11.3.0  PYSEC-2026-3454     12.3.0
pillow         11.3.0  PYSEC-2026-3495     12.3.0
pillow         11.3.0  PYSEC-2026-3496     12.3.0
pillow         11.3.0  PYSEC-2026-3494     12.3.0
pillow         11.3.0  PYSEC-2026-3493     12.3.0
pillow         11.3.0  GHSA-p4wg-p73j-c528 13.0.0
pillow         11.3.0  GHSA-65qm-jvmh-6pqg 13.0.0
pillow         11.3.0  GHSA-58r4-6rv3-gfh8 13.0.0
pillow         11.3.0  GHSA-pchr-8j38-3x4p 13.0.0
pillow         11.3.0  GHSA-q45j-58j2-45rg 13.0.0
pillow         11.3.0  GHSA-462x-pgg6-55fp 13.0.0
pillow         11.3.0  GHSA-c6hw-5grf-ff96 13.0.0
pillow         11.3.0  GHSA-vx6w-vv9f-6gvg 13.0.0
pillow         11.3.0  GHSA-7mwm-ghpf-9vrm 13.0.0
pillow         11.3.0  GHSA-w889-685w-c8fh 13.0.0
pillow         11.3.0  GHSA-j4m9-6w36-pw47 13.0.0
pillow         11.3.0  GHSA-27h8-gf7j-v5hc 13.0.0
pillow         11.3.0  GHSA-fc9g-5qg7-mf45 13.0.0
pillow         11.3.0  GHSA-22hg-r758-rrw9 13.0.0
pillow         11.3.0  GHSA-74r2-7ff9-mjj2 13.0.0
pillow         11.3.0  GHSA-xgh8-jp6q-p5x8 13.0.0
pillow         11.3.0  GHSA-254f-3936-288h 13.0.0
pillow         11.3.0  GHSA-jg86-v3jv-hrpm 13.0.0
pillow         11.3.0  GHSA-f2g6-542v-9j58 13.0.0
pillow         11.3.0  GHSA-5xm2-6676-qcp9 13.0.0
pillow         11.3.0  GHSA-gg2r-wvfm-q9rx 13.0.0
pillow         11.3.0  GHSA-332p-7hrc-92wf 13.0.0
pillow         11.3.0  GHSA-4rr5-89j8-mwwm 13.0.0
pillow         11.3.0  GHSA-24rc-8hj2-v98c 13.0.0
pillow         11.3.0  GHSA-4rm8-3hmm-39fg 13.0.0
pillow         11.3.0  GHSA-mrfp-p49p-3hf2 13.0.0
pillow         11.3.0  GHSA-3j72-p6qg-6xp7 13.0.0
pillow         11.3.0  GHSA-v9pv-5gc8-w2m6 13.0.0
pillow         11.3.0  GHSA-g4qv-2jxx-769h 13.0.0
pillow         11.3.0  GHSA-g842-9rc4-69pw 13.0.0
pillow         11.3.0  GHSA-9v9v-9qvf-h2hj 13.0.0
pillow         11.3.0  GHSA-g7cr-3p24-7wf7 13.0.0
pillow         11.3.0  GHSA-p8hx-v2p6-rp84 13.0.0
pillow         11.3.0  GHSA-78g4-xqwv-q3g8 13.0.0
requests       2.32.5  PYSEC-2026-2275     2.33.0
urllib3        2.6.3   PYSEC-2026-142      2.7.0
urllib3        2.6.3   PYSEC-2026-141      2.7.0
urllib3        2.6.3   PYSEC-2026-4177     2.8.0
urllib3        2.6.3   PYSEC-2026-4176     2.8.0
urllib3        2.6.3   PYSEC-2026-4175     2.8.0
notebook       7.5.8   PYSEC-2026-4112     7.6.3
jupyter-server 2.18.2  PYSEC-2026-366      2.20.0
jupyter-server 2.18.2  PYSEC-2026-4054     2.21.0
anyio          4.12.1  PYSEC-2026-4025     4.14.2
anyio          4.12.1  PYSEC-2026-4024     4.14.2
bleach         6.2.0   GHSA-gj48-438w-jh9v 6.4.0
bleach         6.2.0   GHSA-8rfp-98v4-mmr6 6.4.0
pyarrow        21.0.0  PYSEC-2026-113      23.0.1
soupsieve      2.8.4   PYSEC-2026-4170     2.9.0
soupsieve      2.8.4   PYSEC-2026-4171     2.9.0
```

## Appendix B — Remediation checklist for maintainers

- [x] Commit `models/checksums.json`; verify before every `joblib.load()` (MED-SEC-1)
- [x] `notebooks/build_pipeline.py` regenerates checksums after every retrain (MED-SEC-1)
- [x] Security floors directly in `requirements.txt` (secure default install) + re-audit command (MED-SEC-2)
- [x] Server-side logging instead of `st.exception` (MED-SEC-3)
- [x] `html.escape()` on all dynamic values inside `unsafe_allow_html` blocks (MED-SEC-4)
- [x] HMAC-SHA256 artifact verification with a deployment-supplied key (`MEDADHERE_MODEL_KEY`),
      rejecting artifact+manifest swaps when the key is configured (MED-SEC-1)
- [ ] **Fully close MED-SEC-1:** migrate to a non-executable serialization format (skops/ONNX)
      so a compromise of the whole repository/deployment unit cannot introduce code execution;
      also treat the checksum-only fallback (no key configured) as development-only
- [ ] Optional hardening: run `pip-audit` in CI on every dependency change
- [ ] Optional hardening: pin a lock file (`pip-compile`) for fully reproducible installs
