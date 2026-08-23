"""Property-based tests for the per-provider bearer-token validators.

``_validated_bearer`` is the CWE-20 / CWE-113 / CWE-532 gate every credential passes
before it reaches an ``Authorization`` header. Because the OpenAI and xAI providers keep
independent copies of the validator (D-XAI-002 — no cross-provider imports), both are
exercised here against the same invariants. Every oracle is derived from the RFC 7235
token68 / printable-ASCII contract and the "never echo the credential" security
requirement — never from running the validator itself, so a mutation to the validator's
predicate or its error message diverges from the oracle and is killed.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from claude_bridge.providers.openai.auth import _validated_bearer as openai_validated_bearer
from claude_bridge.providers.xai.auth import _validated_bearer as xai_validated_bearer

_VALIDATORS = [
    pytest.param(openai_validated_bearer, id="openai"),
    pytest.param(xai_validated_bearer, id="xai"),
]

# A realistic credential body: base64url-ish, never a substring of the fixed error
# messages, so the no-leak assertion below cannot match by coincidence.
_TOKEN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

# Codepoints that make a token header-unsafe, each independently rejected by the contract:
# C0 controls (< 0x20) and DEL (0x7F) fail isprintable(); the three >= 0x80 codepoints fail
# isascii(). Built from ordinals so this source stays pure-ASCII and no printable-ASCII
# space (0x20) sneaks in — a space is admitted by isprintable() and would NOT force
# rejection, breaking the leak test below.
_UNSAFE_CODEPOINTS = [0x00, 0x09, 0x0A, 0x0D, 0x1F, 0x7F, 0xE9, 0x2122, 0x2028]
_UNSAFE_CHAR = st.sampled_from([chr(cp) for cp in _UNSAFE_CODEPOINTS])


@pytest.mark.parametrize("validate", _VALIDATORS)
@given(token=st.text())
def test_accepts_iff_nonempty_printable_ascii(validate, token):
    """Accept-and-return-unchanged iff the token is non-empty printable ASCII; else reject.

    Oracle: the RFC 7235 token68 header-safety predicate written independently as an
    explicit codepoint range (U+0020..U+007E), not by calling ``str.isascii`` /
    ``str.isprintable``. A mutant that drops either check diverges on some input.
    """
    spec_accepts = bool(token) and all(0x20 <= ord(c) <= 0x7E for c in token)
    try:
        result = validate(token)
    except ValueError:
        accepted = False
        result = None
    else:
        accepted = True
    assert accepted == spec_accepts
    if accepted:
        assert result == token  # returned unchanged, never rewritten


@pytest.mark.parametrize("validate", _VALIDATORS)
@given(secret=st.text(alphabet=_TOKEN_ALPHABET, min_size=16), bad=_UNSAFE_CHAR)
def test_rejected_credential_is_never_echoed(validate, secret, bad):
    """A malformed credential is rejected AND never appears in the error (CWE-532).

    Oracle: the security requirement itself — the raised message must not contain the
    secret. The 16+ char base64url secret is a plausible real credential; appending any
    header-unsafe char guarantees rejection, so the leak check is always exercised.
    """
    poisoned = secret + bad
    with pytest.raises(ValueError) as excinfo:
        validate(poisoned)
    assert secret not in str(excinfo.value)
