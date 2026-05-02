"""Test helpers for Enigma.

Mirrors ``tilelang.testing.requires_metal``: lets a test compile (emit
MSL) anywhere but skip on hosts without a usable Metal runtime. Useful
for CI on Linux that still exercises trace -> MLIR -> MSL but cannot
launch kernels.
"""

import os
import platform
import sys
import unittest

__all__ = [
    "is_darwin",
    "is_apple_silicon",
    "is_metal_available",
    "requires_metal",
    "requires_apple_silicon",
    "skip_if_no_metal",
]


def is_darwin() -> bool:
    """True iff the host OS is macOS."""
    return sys.platform == "darwin"


def is_apple_silicon() -> bool:
    """True iff the host is macOS on arm64 (M1/M2/M3/M4)."""
    return is_darwin() and platform.machine() == "arm64"


_METAL_AVAILABLE_CACHE: "bool | None" = None


def is_metal_available() -> bool:
    """Best-effort probe for a usable Metal runtime.

    True only when host is macOS AND either ``mlx.core.metal.is_available()``
    returns True or the Enigma Swift dylib reports a Metal device.
    Negative results are cached. Never raises.

    Set ``ENIGMA_FORCE_NO_METAL=1`` to force-disable for testing.
    """
    global _METAL_AVAILABLE_CACHE
    if _METAL_AVAILABLE_CACHE is not None:
        return _METAL_AVAILABLE_CACHE

    if not is_darwin() or os.environ.get("ENIGMA_FORCE_NO_METAL"):
        _METAL_AVAILABLE_CACHE = False
        return False

    try:
        import mlx.core as _mx
        if hasattr(_mx, "metal") and bool(_mx.metal.is_available()):
            _METAL_AVAILABLE_CACHE = True
            return True
    except Exception:
        pass

    try:
        from .runtime_dispatch.runtime import MetalRuntime
        _METAL_AVAILABLE_CACHE = bool(MetalRuntime()._device)
    except Exception:
        _METAL_AVAILABLE_CACHE = False
    return _METAL_AVAILABLE_CACHE


def requires_metal(fn):
    """Skip ``fn`` unless a usable Metal runtime is detected."""
    return unittest.skipUnless(is_metal_available(), "requires Metal runtime")(fn)


def requires_apple_silicon(fn):
    """Skip ``fn`` unless the host is macOS on arm64."""
    return unittest.skipUnless(is_apple_silicon(), "requires Apple Silicon (arm64 macOS)")(fn)


skip_if_no_metal = requires_metal
