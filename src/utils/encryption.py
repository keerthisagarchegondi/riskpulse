"""Field-level encryption utilities for PII protection.

Provides AES-256-GCM encryption/decryption for sensitive fields such as
card numbers, customer names, and other PII. Uses a master key from
environment variables or AWS Secrets Manager.

Security Notes:
- Never log plaintext PII
- Keys should be rotated regularly (90-day policy)
- Use AWS KMS for production key management
"""

from __future__ import annotations

import base64
import os
import secrets
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# Nonce size for AES-GCM (96 bits recommended by NIST)
_NONCE_SIZE = 12

# Key size: 256 bits (32 bytes)
_KEY_SIZE = 32


def _get_encryption_key() -> bytes:
    """Retrieve encryption key from environment.

    In production, this should fetch from AWS Secrets Manager or KMS.
    The key must be exactly 32 bytes (256 bits) base64-encoded.

    Raises:
        ValueError: If key is not configured or invalid length
    """
    key_b64 = os.environ.get("RISKPULSE_ENCRYPTION_KEY")
    if not key_b64:
        raise ValueError(
            "RISKPULSE_ENCRYPTION_KEY environment variable is not set. "
            "Generate one with: python -c \"import secrets, base64; print(base64.b64encode(secrets.token_bytes(32)).decode())\""
        )
    key = base64.b64decode(key_b64)
    if len(key) != _KEY_SIZE:
        raise ValueError(
            f"Encryption key must be exactly {_KEY_SIZE} bytes "
            f"(got {len(key)} bytes). Re-generate with 32 random bytes."
        )
    return key


def generate_key() -> str:
    """Generate a new random encryption key (base64-encoded).

    Returns:
        Base64-encoded 256-bit key suitable for RISKPULSE_ENCRYPTION_KEY
    """
    key = secrets.token_bytes(_KEY_SIZE)
    return base64.b64encode(key).decode("utf-8")


def encrypt_field(plaintext: str, associated_data: Optional[str] = None) -> str:
    """Encrypt a string field using AES-256-GCM.

    Args:
        plaintext: The sensitive value to encrypt
        associated_data: Optional AAD for authenticated encryption
                        (e.g., field name or record ID for binding)

    Returns:
        Base64-encoded ciphertext (nonce || ciphertext || tag)
    """
    if not plaintext:
        return plaintext

    key = _get_encryption_key()
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(_NONCE_SIZE)
    aad = associated_data.encode("utf-8") if associated_data else None

    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), aad)
    # Prepend nonce to ciphertext for storage
    encrypted_blob = nonce + ciphertext
    return base64.b64encode(encrypted_blob).decode("utf-8")


def decrypt_field(encrypted_value: str, associated_data: Optional[str] = None) -> str:
    """Decrypt an AES-256-GCM encrypted field.

    Args:
        encrypted_value: Base64-encoded ciphertext (nonce || ciphertext || tag)
        associated_data: Must match the AAD used during encryption

    Returns:
        Decrypted plaintext string

    Raises:
        cryptography.exceptions.InvalidTag: If ciphertext is tampered or AAD mismatch
    """
    if not encrypted_value:
        return encrypted_value

    key = _get_encryption_key()
    aesgcm = AESGCM(key)
    encrypted_blob = base64.b64decode(encrypted_value)

    nonce = encrypted_blob[:_NONCE_SIZE]
    ciphertext = encrypted_blob[_NONCE_SIZE:]
    aad = associated_data.encode("utf-8") if associated_data else None

    plaintext = aesgcm.decrypt(nonce, ciphertext, aad)
    return plaintext.decode("utf-8")


def mask_field(value: str, visible_chars: int = 4, mask_char: str = "*") -> str:
    """Mask a sensitive field for display (e.g., card numbers in logs).

    Args:
        value: Original value to mask
        visible_chars: Number of trailing characters to keep visible
        mask_char: Character to use for masking

    Returns:
        Masked string, e.g., "****4532"

    Example:
        mask_field("4111111111111234") -> "************1234"
        mask_field("john@example.com", visible_chars=0) -> "****************"
    """
    if not value or len(value) <= visible_chars:
        return mask_char * len(value) if value else ""
    masked_length = len(value) - visible_chars
    return mask_char * masked_length + value[-visible_chars:]


def hash_for_dedup(value: str) -> str:
    """Create a deterministic hash for deduplication without storing PII.

    Uses SHA-256 with a salt derived from the encryption key to prevent
    rainbow table attacks while maintaining deterministic output.

    Args:
        value: Value to hash (e.g., device fingerprint for dedup)

    Returns:
        Hex-encoded SHA-256 hash
    """
    import hashlib

    key = _get_encryption_key()
    # Use first 16 bytes of key as salt
    salt = key[:16]
    return hashlib.sha256(salt + value.encode("utf-8")).hexdigest()
