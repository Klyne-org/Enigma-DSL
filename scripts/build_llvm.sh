#!/usr/bin/env bash
# Build LLVM + MLIR from source with Python bindings enabled.
#
# This is ISOLATED from any Homebrew LLVM install. Nothing here touches
# /opt/homebrew or /usr/local. Everything lives under $HOME/.local/enigma-llvm
# and is activated only when you explicitly source the env file it writes.
#
# Target: Apple Silicon (arm64). Produces a generic arm64 build that works on
# M1, M2, M3, M4 — no -march=native, no CPU-specific tuning.
#
# Usage:
#   ./scripts/build_llvm.sh               # build with defaults
#   ./scripts/build_llvm.sh --clean       # wipe and rebuild from scratch
#   ./scripts/build_llvm.sh --jobs 4      # limit parallelism (default: all cores)
#
# After completion:
#   source $HOME/.local/enigma-llvm/activate.sh
#   # now cmake/llvm/mlir from THIS build take precedence in your shell
#
# To deactivate just open a new terminal — the activation is shell-local.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLVM_VERSION="${LLVM_VERSION:-llvmorg-18.1.8}"   # pinned release tag
PREFIX="${ENIGMA_LLVM_PREFIX:-$HOME/.local/enigma-llvm}"
SRC_DIR="$PREFIX/src/llvm-project"
BUILD_DIR="$PREFIX/build"
INSTALL_DIR="$PREFIX/install"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
CLEAN=0

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) CLEAN=1; shift ;;
    --jobs)  JOBS="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: this script targets macOS" >&2
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "error: this script targets Apple Silicon (arm64)" >&2
  exit 1
fi

for tool in git cmake ninja python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: '$tool' not found. install via: brew install cmake ninja git" >&2
    exit 1
  fi
done

# Use Apple's system clang/clang++ so we DON'T accidentally pick up
# a Homebrew LLVM's clang (which would create a dependency between the
# two installs). xcrun always resolves to Apple's Command Line Tools.
CC="$(xcrun --find clang)"
CXX="$(xcrun --find clang++)"
SDKROOT="$(xcrun --show-sdk-path)"
export CC CXX SDKROOT

# Pick a Python. MLIR 18.1.8 bindings need Python 3.10–3.12 (3.13+ is too new).
# We create a dedicated venv under $PREFIX so we don't fight Homebrew's
# PEP 668 restriction and don't pollute the system Python.
VENV_DIR="$PREFIX/venv"

pick_python() {
  for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$(command -v "$candidate")"
      return 0
    fi
  done
  # Fallback: system python3 only if version is in range
  local v
  v="$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null || echo 0)"
  if [[ "$v" -ge 310 && "$v" -le 312 ]]; then
    echo "$(command -v python3)"
    return 0
  fi
  return 1
}

HOST_PYTHON="$(pick_python)" || {
  echo "error: need Python 3.10, 3.11, or 3.12 for MLIR bindings." >&2
  echo "       install via: brew install python@3.12" >&2
  exit 1
}

if [[ ! -d "$VENV_DIR" ]]; then
  echo "creating venv at $VENV_DIR using $HOST_PYTHON"
  "$HOST_PYTHON" -m venv "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python3"
PYTHON_VERSION="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

"$PYTHON" -m pip install --upgrade pip wheel setuptools
# Pin pybind11 to 2.13.x — MLIR 18.1.8 uses def_property with keep_alive,
# which was removed in pybind11 3.x. Same reasoning for nanobind.
"$PYTHON" -m pip install "pybind11<3" "nanobind<2" numpy PyYAML

echo "──────────────────────────────────────────────────────────────"
echo " LLVM version : $LLVM_VERSION"
echo " Prefix       : $PREFIX"
echo " Python       : $PYTHON ($PYTHON_VERSION)"
echo " CC/CXX       : $CC / $CXX"
echo " SDK          : $SDKROOT"
echo " Jobs         : $JOBS"
echo "──────────────────────────────────────────────────────────────"

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

if [[ $CLEAN -eq 1 ]]; then
  echo "cleaning $PREFIX"
  rm -rf "$BUILD_DIR" "$INSTALL_DIR"
fi

mkdir -p "$PREFIX/src"

# ---------------------------------------------------------------------------
# Fetch source (shallow clone at pinned tag)
# ---------------------------------------------------------------------------

if [[ ! -d "$SRC_DIR/.git" ]]; then
  echo "cloning llvm-project @ $LLVM_VERSION"
  git clone --depth 1 --branch "$LLVM_VERSION" \
    https://github.com/llvm/llvm-project.git "$SRC_DIR"
else
  echo "source already present at $SRC_DIR"
  (cd "$SRC_DIR" && git fetch --depth 1 origin tag "$LLVM_VERSION" && git checkout "$LLVM_VERSION")
fi

# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------

mkdir -p "$BUILD_DIR"

# Keep the config minimal:
#   - only MLIR project (not clang/lldb/lld — saves ~2x build time)
#   - only AArch64 target (Metal runs on arm64, we don't need X86/NVPTX)
#   - bindings enabled
#   - release build with assertions (catches dialect bugs without huge slowdown)
#   - explicit install prefix so nothing escapes $PREFIX
cmake -S "$SRC_DIR/llvm" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_OSX_SYSROOT="$SDKROOT" \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="AArch64" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_LINK_LLVM_DYLIB=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$PYTHON" \
  -DLLVM_PARALLEL_LINK_JOBS=2

# ---------------------------------------------------------------------------
# Build + install
# ---------------------------------------------------------------------------

echo "building (this takes 30-90 minutes the first time)…"
ninja -C "$BUILD_DIR" -j "$JOBS"

echo "installing to $INSTALL_DIR"
ninja -C "$BUILD_DIR" install

# ---------------------------------------------------------------------------
# Write activation script
# ---------------------------------------------------------------------------

cat > "$PREFIX/activate.sh" <<EOF
# Source this file to put the Enigma LLVM build first on PATH and activate
# the dedicated Python venv. Does NOT modify your global shell config.
export ENIGMA_LLVM_PREFIX="$PREFIX"
export ENIGMA_LLVM_INSTALL="$INSTALL_DIR"
export PATH="\$ENIGMA_LLVM_INSTALL/bin:$VENV_DIR/bin:\$PATH"
export MLIR_DIR="\$ENIGMA_LLVM_INSTALL/lib/cmake/mlir"
export LLVM_DIR="\$ENIGMA_LLVM_INSTALL/lib/cmake/llvm"
export DYLD_LIBRARY_PATH="\$ENIGMA_LLVM_INSTALL/lib:\${DYLD_LIBRARY_PATH:-}"
export PYTHONPATH="\$ENIGMA_LLVM_INSTALL/python_packages/mlir_core:\${PYTHONPATH:-}"
export VIRTUAL_ENV="$VENV_DIR"
echo "activated Enigma LLVM at \$ENIGMA_LLVM_INSTALL"
echo "python: \$(which python3)"
EOF
chmod +x "$PREFIX/activate.sh"

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

echo ""
echo "──────────────────────────────────────────────────────────────"
echo " Verifying the build…"
echo "──────────────────────────────────────────────────────────────"

# shellcheck disable=SC1091
source "$PREFIX/activate.sh"

echo -n "  llvm-config: "; "$INSTALL_DIR/bin/llvm-config" --version
echo -n "  mlir-opt:    "; "$INSTALL_DIR/bin/mlir-opt" --version | head -n1

# The critical test: do the Python bindings import?
if "$PYTHON" -c "from mlir import ir; ctx = ir.Context(); print('mlir.ir OK')"; then
  echo "  python bindings: OK"
else
  echo "  python bindings: FAILED" >&2
  exit 1
fi

echo ""
echo "──────────────────────────────────────────────────────────────"
echo " Build complete."
echo ""
echo " To use this LLVM in a shell:"
echo "     source $PREFIX/activate.sh"
echo ""
echo " Key exported vars after activation:"
echo "     MLIR_DIR, LLVM_DIR, PATH (with llvm bin first),"
echo "     PYTHONPATH (with mlir_core bindings first)"
echo "──────────────────────────────────────────────────────────────"
