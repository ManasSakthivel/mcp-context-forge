# -*- coding: utf-8 -*-
"""Location: ./mcpgateway/common/oauth.py
Copyright contributors to the MCP-CONTEXT-FORGE project
SPDX-License-Identifier: Apache-2.0

Shared OAuth helpers and sensitive-key definitions.
"""

# Standard
from typing import Any, Optional

# Third-Party
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

# OAuth config keys that should always be treated as secret material.
OAUTH_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "client_secret",
        "password",
        "refresh_token",
        "access_token",
        "id_token",
        "token",
        "secret",
        "private_key",
    }
)

# Token endpoint client authentication methods honored at runtime (RFC 6749
# Section 2.3.1, RFC 7591 Section 2.3, RFC 7523 Section 2.2). The schema and
# the runtime read this same set so a config value rejected at the API
# boundary can never be silently honored with a different method later.
SUPPORTED_TOKEN_ENDPOINT_AUTH_METHODS: frozenset[str] = frozenset(
    {
        "none",
        "client_secret_basic",
        "client_secret_post",
        "private_key_jwt",
    }
)

# client_assertion_type value for JWT bearer client assertions (RFC 7523
# Section 2.2).
CLIENT_ASSERTION_TYPE_JWT_BEARER: str = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"

# Asymmetric signing algorithms allowed for private_key_jwt client
# assertions. Symmetric HS* and the "none" algorithm are intentionally
# excluded: the gateway signs outbound assertions with asymmetric key
# material and must never fall back to a shared-secret or unsigned mode.
SUPPORTED_TOKEN_ENDPOINT_SIGNING_ALGS: frozenset[str] = frozenset(
    {
        "RS256",
        "RS384",
        "RS512",
        "ES256",
        "ES384",
        "ES512",
        "PS256",
    }
)

DEFAULT_TOKEN_ENDPOINT_SIGNING_ALG: str = "RS256"

# Upper bound for the assertion exp-iat window (RFC 7523 recommends a short
# lifetime; providers reject assertions that live too long).
MAX_CLIENT_ASSERTION_TTL_SECONDS: int = 300

# RSA-backed assertion signing algorithms (RFC 7518 Section 3.3 and 3.5).
_RSA_SIGNING_ALGS: frozenset[str] = frozenset({"RS256", "RS384", "RS512", "PS256"})

# EC-backed assertion signing algorithms mapped to the exact curve JWA requires
# (RFC 7518 Section 3.4). PyJWT rejects a curve mismatch at sign time, so the
# same constraint is enforced at configuration time. Note ES512 pairs with
# P-521, not P-512.
_EC_SIGNING_ALG_CURVES: dict[str, type] = {
    "ES256": ec.SECP256R1,
    "ES384": ec.SECP384R1,
    "ES512": ec.SECP521R1,
}


def is_sensitive_oauth_key(key: Any) -> bool:
    """Return whether an oauth_config key should be treated as secret.

    Args:
        key: Candidate oauth_config key.

    Returns:
        bool: True when key maps to sensitive OAuth material.
    """
    return isinstance(key, str) and key.lower() in OAUTH_SENSITIVE_KEYS


def validate_private_key_material(pem: str, alg: Optional[str] = None) -> None:
    """Validate raw PEM private key material against its signing algorithm.

    Deliberately pure: it takes already-resolved plaintext PEM and imports nothing
    from the encryption service, so ``mcpgateway.common.oauth`` stays free of the
    import cycle that ``encryption_service`` would create. Callers decide whether a
    value is raw material at all; masked placeholders and stored ciphertext must be
    filtered out before this runs.

    Args:
        pem: Plaintext PEM-encoded private key material.
        alg: Requested ``token_endpoint_auth_signing_alg``. Defaults to
            ``DEFAULT_TOKEN_ENDPOINT_SIGNING_ALG`` when omitted. Matched
            case-sensitively, the same way the schema validator and the signing
            path in ``OAuthManager`` match it, so a value accepted here cannot be
            rejected later (or the reverse).

    Raises:
        ValueError: If the material is not a parseable private key, is
            passphrase-protected, or does not match the requested algorithm.
            Messages never include the supplied value, which is
            attacker-influenced and reaches logs.
    """
    if not isinstance(pem, str) or not pem.strip():
        raise ValueError("oauth_config.private_key must be a non-empty string when token_endpoint_auth_method is private_key_jwt")

    alg_name = alg or DEFAULT_TOKEN_ENDPOINT_SIGNING_ALG
    if alg_name not in SUPPORTED_TOKEN_ENDPOINT_SIGNING_ALGS:
        raise ValueError(f"oauth_config.token_endpoint_auth_signing_alg is not allowed. Supported values: {', '.join(sorted(SUPPORTED_TOKEN_ENDPOINT_SIGNING_ALGS))}")

    try:
        private_key = serialization.load_pem_private_key(pem.encode(), password=None)
    except TypeError as exc:
        # cryptography raises TypeError when the PEM is encrypted and no password
        # is supplied. PyJWT cannot sign with such a key either, so reject it here
        # rather than failing on the first outbound token request.
        raise ValueError("oauth_config.private_key is passphrase-protected; supply an unencrypted private key for private_key_jwt") from exc
    except (ValueError, UnsupportedAlgorithm) as exc:
        # Covers malformed PEM, public-key PEM, and certificate PEM. The original
        # message can echo the input, so it is deliberately not interpolated.
        raise ValueError("oauth_config.private_key is not a valid PEM-encoded private key") from exc

    if alg_name in _RSA_SIGNING_ALGS:
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise ValueError(f"oauth_config.private_key must be an RSA private key for token_endpoint_auth_signing_alg {alg_name}")
        return

    expected_curve = _EC_SIGNING_ALG_CURVES[alg_name]
    if not isinstance(private_key, ec.EllipticCurvePrivateKey):
        raise ValueError(f"oauth_config.private_key must be an EC private key for token_endpoint_auth_signing_alg {alg_name}")
    if not isinstance(private_key.curve, expected_curve):
        raise ValueError(f"oauth_config.private_key curve does not match token_endpoint_auth_signing_alg {alg_name}; expected {expected_curve.name}")
