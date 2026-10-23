"""
Model Artifact Integrity Manifest Generator
===========================================
Regenerates models/checksums.json with SHA-256 hashes of the serialized
scikit-learn artifacts (joblib pickle files).

Run this AFTER every retraining (models are not byte-reproducible across
runs), then commit the updated manifest together with the new artifacts:

    python models/update_checksums.py

Deployment hardening: if the environment variable MEDADHERE_MODEL_KEY is set,
the manifest additionally carries per-artifact HMAC-SHA256 digests keyed with
that secret. The Streamlit app then verifies the HMAC before joblib.load(),
which rejects tampering even when both the artifact and the manifest are
swapped (the secret lives only in the deployment environment, never in the
repository):

    MEDADHERE_MODEL_KEY=<secret> python models/update_checksums.py
    # and set the same variable where the app runs
"""

import hashlib
import hmac
import json
import os

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKSUMS_PATH = os.path.join(MODELS_DIR, 'checksums.json')
MODEL_KEY_ENV = 'MEDADHERE_MODEL_KEY'

# Artifacts that are deserialized with joblib.load() by the application.
PROTECTED_ARTIFACTS = [
    'medication_adherence_model.pkl',
    'preprocessing_pipeline.pkl',
]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def hmac_of(key, path):
    h = hmac.new(key, digestmod=hashlib.sha256)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    key = os.environ.get(MODEL_KEY_ENV, '').encode() or None
    artifacts = {}
    for name in PROTECTED_ARTIFACTS:
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model artifact not found: {path}. Train the model first."
            )
        entry = {'sha256': sha256_of(path), 'hmac': hmac_of(key, path) if key else None}
        artifacts[f'models/{name}'] = entry

    manifest = {'schema': 2, 'artifacts': artifacts}
    with open(CHECKSUMS_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')

    print(f"Updated {CHECKSUMS_PATH} "
          f"(HMAC entries: {'yes' if key else 'no — set ' + MODEL_KEY_ENV + ' for deployment signing'}):")
    for artifact, entry in artifacts.items():
        print(f"  sha256={entry['sha256']}  {artifact}")


if __name__ == '__main__':
    main()
