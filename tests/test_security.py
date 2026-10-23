"""
Security regression tests: model artifact integrity verification (MED-SEC-1).

Run with the repository root as working directory:
    python -m unittest discover -s tests -v

No third-party dependencies: app/integrity.py uses only the standard library.
"""

import hashlib
import hmac
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

import integrity  # noqa: E402


def _write_manifest(path, artifacts):
    with open(path, 'w') as f:
        json.dump({'schema': 2, 'artifacts': artifacts}, f)


class ArtifactIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        os.makedirs(os.path.join(self.tmp, 'models'))
        self.artifact = os.path.join(self.tmp, 'models', 'a.pkl')
        Path(self.artifact).write_bytes(b'\x80\x04\x95fake-pickle-payload')

    def _manifest_path(self):
        return os.path.join(self.tmp, 'checksums.json')

    def _pristine_sha(self):
        return hashlib.sha256(Path(self.artifact).read_bytes()).hexdigest()

    # ---- SHA-256 tier (no MEDADHERE_MODEL_KEY) ----

    def test_pristine_artifact_passes(self):
        _write_manifest(self._manifest_path(),
                        {'models/a.pkl': {'sha256': self._pristine_sha(), 'hmac': None}})
        integrity.verify_artifact_integrity(self.artifact, self.tmp, self._manifest_path())

    def test_missing_manifest_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            integrity.verify_artifact_integrity(self.artifact, self.tmp, self._manifest_path())

    def test_unregistered_artifact_rejected(self):
        _write_manifest(self._manifest_path(), {})
        with self.assertRaisesRegex(RuntimeError, 'No checksum registered'):
            integrity.verify_artifact_integrity(self.artifact, self.tmp, self._manifest_path())

    def test_tampered_artifact_rejected(self):
        _write_manifest(self._manifest_path(),
                        {'models/a.pkl': {'sha256': '0' * 64, 'hmac': None}})
        with open(self.artifact, 'r+b') as f:
            f.seek(10)
            byte = f.read(1)
            f.seek(10)
            f.write(bytes([byte[0] ^ 0xFF]))
        with self.assertRaisesRegex(RuntimeError, 'Integrity check failed'):
            integrity.verify_artifact_integrity(self.artifact, self.tmp, self._manifest_path())

    # ---- HMAC tier (MEDADHERE_MODEL_KEY set) ----

    KEY = b'deployment-secret'

    def setUpHmac(self, entry):
        os.environ[integrity.MODEL_KEY_ENV] = self.KEY.decode()
        self.addCleanup(os.environ.pop, integrity.MODEL_KEY_ENV, None)
        _write_manifest(self._manifest_path(), {'models/a.pkl': entry})

    def test_valid_hmac_passes(self):
        digest = hmac.new(self.KEY, Path(self.artifact).read_bytes(), hashlib.sha256).hexdigest()
        self.setUpHmac({'sha256': self._pristine_sha(), 'hmac': digest})
        integrity.verify_artifact_integrity(self.artifact, self.tmp, self._manifest_path())

    def test_manifest_and_artifact_swap_rejected(self):
        """The attack the in-repo checksum alone cannot stop: an attacker replaces the
        artifact AND regenerates the manifest (valid sha256). Without the deployment
        key they cannot forge the HMAC, so verification must refuse the load."""
        Path(self.artifact).write_bytes(b'\x80\x04\x95attacker-controlled-pickle')
        attacker_sha = hashlib.sha256(Path(self.artifact).read_bytes()).hexdigest()
        # Attacker-updated manifest: internally consistent, but HMAC cannot be forged.
        self.setUpHmac({'sha256': attacker_sha, 'hmac': 'a' * 64})
        with self.assertRaisesRegex(RuntimeError, 'HMAC verification failed'):
            integrity.verify_artifact_integrity(self.artifact, self.tmp, self._manifest_path())

    def test_hmac_required_but_missing_in_manifest_fails_closed(self):
        self.setUpHmac({'sha256': self._pristine_sha(), 'hmac': None})
        with self.assertRaisesRegex(RuntimeError, 'no HMAC is registered'):
            integrity.verify_artifact_integrity(self.artifact, self.tmp, self._manifest_path())


if __name__ == '__main__':
    unittest.main()
