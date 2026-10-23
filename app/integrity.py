"""
Model Artifact Integrity Verification (MedAdhere AI)
====================================================
Joblib/pickle files execute arbitrary code on load, so every model artifact
is verified BEFORE deserialization in app.py.

Two verification tiers, controlled by the MEDADHERE_MODEL_KEY environment
variable:

- Key NOT set:  SHA-256 checksum verification against models/checksums.json.
                Catches accidental corruption and partial uploads. The
                manifest lives in the same repository as the artifacts, so it
                is NOT an independent trust anchor against a committed
                artifact+manifest swap.
- Key set:      HMAC-SHA256 verification using the deployment-supplied secret,
                in addition to the SHA-256 check. An attacker who can swap
                both the artifact and the manifest still cannot forge the
                HMAC without the key, so this tier rejects committed
                tampering. Fails closed: if the key is set but the manifest
                lacks HMAC entries, the load is refused.

Generate/refresh the manifest with:  python models/update_checksums.py
(set MEDADHERE_MODEL_KEY in the deployment environment to also emit HMACs).
"""

import hashlib
import hmac
import json
import logging
import os

logger = logging.getLogger(__name__)

MODEL_KEY_ENV = "MEDADHERE_MODEL_KEY"


def _sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _hmac_of(key, path):
    h = hmac.new(key, digestmod=hashlib.sha256)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_artifact_integrity(artifact_path, base_dir, checksums_path):
    """
    Verify a joblib artifact against models/checksums.json before unpickling.

    Raises RuntimeError on any failure; returns normally only if the artifact
    is safe to deserialize under the active verification tier.
    """
    if not os.path.exists(checksums_path):
        raise RuntimeError(
            "Model integrity manifest (models/checksums.json) is missing. "
            "Refusing to load unverified model artifacts. "
            "Run 'python models/update_checksums.py' after training."
        )
    with open(checksums_path, 'r') as f:
        manifest = json.load(f)

    artifacts = manifest.get('artifacts')
    if not isinstance(artifacts, dict):
        raise RuntimeError(
            "Model integrity manifest has an unrecognized format. "
            "Regenerate it with 'python models/update_checksums.py'."
        )

    repo_relative = os.path.relpath(artifact_path, base_dir).replace(os.sep, '/')
    if repo_relative not in artifacts:
        raise RuntimeError(
            f"No checksum registered for '{repo_relative}'. "
            "Run 'python models/update_checksums.py' after training."
        )
    entry = artifacts[repo_relative]

    key = os.environ.get(MODEL_KEY_ENV, '').encode() or None
    if key is not None:
        # Deployment tier: HMAC with an out-of-repo secret. Fails closed when
        # the manifest carries no HMAC for this artifact.
        expected_hmac = (entry or {}).get('hmac')
        if not expected_hmac:
            raise RuntimeError(
                f"HMAC verification required (environment variable "
                f"{MODEL_KEY_ENV} is set) but no HMAC is registered for "
                f"'{repo_relative}'. Regenerate the manifest with the same "
                "key: MEDADHERE_MODEL_KEY=<key> python models/update_checksums.py"
            )
        if not hmac.compare_digest(_hmac_of(key, artifact_path), expected_hmac):
            raise RuntimeError(
                f"HMAC verification failed for '{repo_relative}'. The artifact "
                "does not match the deployment-signed manifest and will not "
                "be loaded."
            )
        logger.info("HMAC verification passed for %s", repo_relative)
        return

    # Local/development tier: plain checksum. Catches corruption; the manifest
    # is not an independent trust anchor against committed tampering.
    logger.warning(
        "%s is not set; falling back to plain SHA-256 checksum verification "
        "of model artifacts (no protection against committed tampering).",
        MODEL_KEY_ENV,
    )
    expected_sha = (entry or {}).get('sha256')
    if not expected_sha:
        raise RuntimeError(
            f"No SHA-256 checksum registered for '{repo_relative}'. "
            "Run 'python models/update_checksums.py' after training."
        )
    if _sha256_of(artifact_path) != expected_sha:
        raise RuntimeError(
            f"Integrity check failed for '{repo_relative}'. The model artifact "
            "does not match its committed checksum and will not be loaded. "
            "Restore the trusted artifact or retrain the model."
        )
