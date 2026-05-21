#!/usr/bin/env bash
# Unified build pipeline for Enigma (DSL + Dialect).
#
# Builds ONE self-contained wheel per Python version that includes both the
# DSL and the dialect (C++/MLIR bindings + dylibs). Drops it in ./wheelhouse/,
# and (by default) installs it into a per-Python venv and runs the test suite.
#
# Usage:
#   ./build_all.sh                                # Python 3.12 (default), install + test in Enigma-DSL/.venv
#   ./build_all.sh --python 3.12 --python 3.13    # multi-version
#   ./build_all.sh --no-test                      # build + install, skip pytest
#   ./build_all.sh --no-install                   # build wheels only
#   ./build_all.sh --no-merge                     # keep two separate wheels
#   ./build_all.sh --skip-dialect                 # reuse pre-built dialect wheel
#   ./build_all.sh --skip-dsl                     # only build the dialect
#   ./build_all.sh --clean                        # wipe build/ caches first
#   ./build_all.sh --out /tmp/wheels              # custom wheelhouse directory
#
# Output (default, --merge):
#   ./wheelhouse/enigma-<ver>-cpXY-cpXY-macosx_*_arm64.whl
#       -> single merged wheel containing both `enigma/` and `mlir/` (dialect)
#
# Output with --no-merge:
#   ./wheelhouse/enigma_dsl-*.whl                     # pure-Python DSL wheel
#   ./wheelhouse/enigma_dialect-*-cpXY-cpXY-*.whl # one per Python ABI
#
# Requires:
#   - LLVM/MLIR pre-built at $HOME/.local/enigma-llvm (run scripts/build_llvm.sh once)
#   - Apple Silicon macOS, Xcode CLT
#   - Each requested python<X.Y> must be on PATH or in /opt/homebrew/bin

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIALECT_DIR="$ROOT/Enigma-Dialect"
DSL_DIR="$ROOT"
WHEELHOUSE_DEFAULT="$ROOT/wheelhouse"

PY_VERSIONS=()
MACOS_TARGETS=()
DO_TEST=1
DO_INSTALL=1
SKIP_DIALECT=0
SKIP_DSL=0
DO_CLEAN=0
DO_MERGE=1
WHEELHOUSE="$WHEELHOUSE_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)       PY_VERSIONS+=("$2"); shift 2 ;;
    --macos)        MACOS_TARGETS+=("$2"); shift 2 ;;
    --no-test)      DO_TEST=0; shift ;;
    --no-install)   DO_INSTALL=0; shift ;;
    --no-merge)     DO_MERGE=0; shift ;;
    --skip-dialect) SKIP_DIALECT=1; shift ;;
    --skip-dsl)     SKIP_DSL=1; shift ;;
    --clean)        DO_CLEAN=1; shift ;;
    --out)          WHEELHOUSE="$2"; shift 2 ;;
    -h|--help)      sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ ${#PY_VERSIONS[@]} -eq 0 ]] && PY_VERSIONS=(3.12)
# Default macOS target: derive from current OS major (e.g. Darwin 24 -> macOS 15.0,
# Darwin 23 -> macOS 14.0). If a list is passed via --macos, use it verbatim.
if [[ ${#MACOS_TARGETS[@]} -eq 0 ]]; then
  _DARWIN_MAJ="$(uname -r | cut -d. -f1)"
  case "$_DARWIN_MAJ" in
    24) MACOS_TARGETS=(15.0) ;;
    23) MACOS_TARGETS=(14.0) ;;
    *)  MACOS_TARGETS=("$(sw_vers -productVersion | cut -d. -f1).0") ;;
  esac
fi

# --- Locate Python interpreters (bash 3.2 compat: no associative arrays) ---
resolve_python() {
  local V="$1"
  for cand in "python$V" "/opt/homebrew/bin/python$V" "/usr/local/bin/python$V" \
              "/Library/Frameworks/Python.framework/Versions/$V/bin/python$V"; do
    if command -v "$cand" >/dev/null 2>&1; then
      command -v "$cand"
      return 0
    fi
  done
  return 1
}

for V in "${PY_VERSIONS[@]}"; do
  if ! resolve_python "$V" >/dev/null; then
    echo "error: python$V not found on this system" >&2
    exit 1
  fi
done

echo "============================================================"
echo "Enigma unified build"
echo "  Python versions : ${PY_VERSIONS[*]}"
echo "  macOS targets   : ${MACOS_TARGETS[*]}"
echo "  Wheelhouse      : $WHEELHOUSE"
echo "  Build dialect   : $([[ $SKIP_DIALECT -eq 0 ]] && echo yes || echo no)"
echo "  Build dsl       : $([[ $SKIP_DSL     -eq 0 ]] && echo yes || echo no)"
echo "  Merge into one  : $([[ $DO_MERGE     -eq 1 ]] && echo yes || echo no)"
echo "  Install + test  : install=$DO_INSTALL test=$DO_TEST"
echo "============================================================"

mkdir -p "$WHEELHOUSE"

# --- Activate LLVM/MLIR env (only needed for dialect) ----------------------
if [[ $SKIP_DIALECT -eq 0 ]]; then
  if [[ -z "${MLIR_DIR:-}" ]]; then
    if [[ -f "$HOME/.local/enigma-llvm/activate.sh" ]]; then
      # shellcheck disable=SC1091
      source "$HOME/.local/enigma-llvm/activate.sh"
    else
      echo "error: MLIR_DIR not set and ~/.local/enigma-llvm/activate.sh missing." >&2
      echo "       Run Enigma-Dialect/scripts/build_llvm.sh first."             >&2
      exit 1
    fi
  fi
fi

# --- Helpers ----------------------------------------------------------------
fix_dialect_rpaths() {
  local WHL="$1"
  local TMP
  TMP=$(mktemp -d)
  python3 -m wheel unpack -d "$TMP" "$WHL" >/dev/null
  local UNPACKED PKG_DIR
  UNPACKED=$(ls -d "$TMP"/enigma_dialect-*/ | head -1)
  PKG_DIR="${UNPACKED}mlir/_mlir_libs"

  for f in "$PKG_DIR"/*.dylib "$PKG_DIR"/*.so; do
    [[ -e "$f" ]] || continue
    for dep in $(otool -L "$f" | awk 'NR>1 {print $1}'); do
      case "$dep" in
        /opt/homebrew/*libzstd*|/usr/local/*libzstd*)
          install_name_tool -change "$dep" "@loader_path/$(basename "$dep")" "$f" 2>/dev/null || true ;;
      esac
    done
    codesign --force --sign - "$f" 2>/dev/null || true
  done

  for ext in "$PKG_DIR"/_mlir*.so; do
    [[ -e "$ext" ]] || continue
    if ! otool -l "$ext" | grep -q '@loader_path'; then
      install_name_tool -add_rpath '@loader_path' "$ext" 2>/dev/null || true
    fi
  done

  rm -f "$WHL"
  python3 -m wheel pack -d "$(dirname "$WHL")" "$UNPACKED" >/dev/null
  rm -rf "$TMP"
}

build_dialect_for() {
  local V="$1"
  local MAC="${2:-}"
  local PY_BIN
  PY_BIN="$(resolve_python "$V")"
  echo
  echo "==> [dialect] Python $V ($PY_BIN) macOS=$MAC"

  cd "$DIALECT_DIR"
  if [[ $DO_CLEAN -eq 1 ]]; then
    rm -rf build-wheel dist
  else
    rm -rf dist
  fi

  "$PY_BIN" -m pip install --quiet --upgrade build scikit-build-core wheel

  # Temporarily relax the requires-python pin so non-3.12 Pythons can build.
  local PYPROJECT="$DIALECT_DIR/pyproject.toml"
  local BACKUP="$PYPROJECT.bak.$$"
  cp "$PYPROJECT" "$BACKUP"
  python3 - "$PYPROJECT" "$V" <<'PY'
import re, sys
path, ver = sys.argv[1], sys.argv[2]
s = open(path).read()
s = re.sub(r'requires-python\s*=\s*"[^"]*"',
           f'requires-python = "=={ver}.*"', s)
open(path, "w").write(s)
PY

  local MAC_ENV=()
  if [[ -n "$MAC" ]]; then
    MAC_ENV=(
      MACOSX_DEPLOYMENT_TARGET="$MAC"
      _PYTHON_HOST_PLATFORM="macosx-${MAC}-arm64"
      ARCHFLAGS="-arch arm64"
    )
  fi

  env "${MAC_ENV[@]}" \
    CC="${CC:-/usr/bin/clang}" CXX="${CXX:-/usr/bin/clang++}" \
    "$PY_BIN" -m build --wheel --no-isolation
  mv "$BACKUP" "$PYPROJECT"

  local WHL
  WHL=$(ls "$DIALECT_DIR"/dist/enigma_dialect-*.whl | head -1)
  fix_dialect_rpaths "$WHL"
  WHL=$(ls "$DIALECT_DIR"/dist/enigma_dialect-*.whl | head -1)
  cp "$WHL" "$WHEELHOUSE/"
  echo "    -> $(basename "$WHL")"
}

build_dsl() {
  echo
  echo "==> [dsl] (pure-Python, version-agnostic wheel)"

  cd "$DSL_DIR"
  if [[ $DO_CLEAN -eq 1 ]]; then
    rm -rf build dist
  else
    rm -rf dist
  fi

  # Use whichever python is on PATH; the resulting wheel is py3-none-any.
  python3 -m pip install --quiet --upgrade build setuptools wheel setuptools-scm
  python3 -m build --wheel --no-isolation
  cp "$DSL_DIR"/dist/enigma_dsl-*.whl "$WHEELHOUSE/"
  echo "    -> $(basename "$(ls "$DSL_DIR"/dist/enigma_dsl-*.whl | head -1)")"
}

ensure_venv_for() {
  local V="$1"
  local PY_BIN
  PY_BIN="$(resolve_python "$V")"
  local VENV
  if [[ "$V" == "3.12" ]]; then
    VENV="$DSL_DIR/.venv"
  else
    VENV="$DSL_DIR/.venv-py$V"
  fi
  if [[ ! -x "$VENV/bin/python" ]]; then
    echo "    creating venv at $VENV"
    "$PY_BIN" -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
  fi
  echo "$VENV"
}

merge_wheels_for() {
  local V="$1"
  local MAC="${2:-}"
  local PY_TAG="cp${V//./}"
  local MAC_GLOB="*"
  [[ -n "$MAC" ]] && MAC_GLOB="macosx_${MAC//./_}_arm64"
  echo
  echo "==> Merging DSL + dialect into single 'enigma' wheel for Python $V macOS=$MAC"

  local DSL_WHL DIALECT_WHL
  DSL_WHL=$(ls "$WHEELHOUSE"/enigma_dsl-*-py3-none-any.whl 2>/dev/null | head -1 || true)
  DIALECT_WHL=$(ls "$WHEELHOUSE"/enigma_dialect-*-${PY_TAG}-${PY_TAG}-${MAC_GLOB}.whl 2>/dev/null | head -1 || true)

  if [[ -z "$DSL_WHL" || -z "$DIALECT_WHL" ]]; then
    echo "    cannot merge: need both enigma_dsl-*-py3-none-any.whl and enigma_dialect-${PY_TAG}-${PY_TAG}-${MAC_GLOB}.whl"
    return 1
  fi

  local TMP
  TMP=$(mktemp -d)
  trap "rm -rf '$TMP'" RETURN

  python3 -m wheel unpack -d "$TMP/dsl"     "$DSL_WHL"     >/dev/null
  python3 -m wheel unpack -d "$TMP/dialect" "$DIALECT_WHL" >/dev/null

  local DSL_DIR DIA_DIR
  DSL_DIR=$(ls -d "$TMP/dsl"/enigma_dsl-*/      | head -1)
  DIA_DIR=$(ls -d "$TMP/dialect"/enigma_dialect-*/ | head -1)

  cp -R "$DIA_DIR/mlir" "$DSL_DIR/"

  local DSL_DIST_INFO DIA_TAG
  DSL_DIST_INFO=$(ls -d "$DSL_DIR"*.dist-info | head -1)
  DIA_TAG=$(awk '/^Tag: /{print $2; exit}' "$DIA_DIR"*.dist-info/WHEEL)

  python3 - "$DSL_DIST_INFO/WHEEL" "$DIA_TAG" <<'PY'
import sys
path, tag = sys.argv[1], sys.argv[2]
out = []
for line in open(path).read().splitlines():
    if line.startswith("Tag:"):
        continue
    if line.startswith("Root-Is-Purelib:"):
        line = "Root-Is-Purelib: false"
    out.append(line)
while out and not out[-1].strip():
    out.pop()
out.append(f"Tag: {tag}")
open(path, "w").write("\n".join(out) + "\n")
PY

  # Only delete the per-Python dialect wheel (consumed); keep the pure-Python
  # DSL wheel so subsequent matrix iterations can reuse it.
  rm -f "$DIALECT_WHL"
  python3 -m wheel pack -d "$WHEELHOUSE" "$DSL_DIR" >/dev/null
  local MERGED
  MERGED=$(ls -t "$WHEELHOUSE"/enigma_dsl-*${DIA_TAG}*.whl | head -1)
  echo "    -> $(basename "$MERGED")"
}

install_into() {
  local VENV="$1"
  local V="${2:-}"
  local MAC="${3:-}"
  local PY_TAG=""
  [[ -n "$V" ]] && PY_TAG="cp${V//./}"
  local MAC_GLOB="*"
  [[ -n "$MAC" ]] && MAC_GLOB="macosx_${MAC//./_}_arm64"
  echo
  echo "==> Installing wheels into $VENV (macOS=$MAC)"
  local PKGS=()
  if [[ $DO_MERGE -eq 1 ]]; then
    local MERGED
    if [[ -n "$PY_TAG" ]]; then
      MERGED=$(ls "$WHEELHOUSE"/enigma_dsl-*-${PY_TAG}-${PY_TAG}-${MAC_GLOB}.whl 2>/dev/null | head -1 || true)
    else
      MERGED=$(ls "$WHEELHOUSE"/enigma_dsl-*-cp*-cp*-${MAC_GLOB}.whl 2>/dev/null | head -1 || true)
    fi
    [[ -n "$MERGED" ]] && PKGS+=("$MERGED")
  else
    local DIALECT_WHL DSL_WHL
    if [[ -n "$PY_TAG" ]]; then
      DIALECT_WHL=$(ls "$WHEELHOUSE"/enigma_dialect-*-${PY_TAG}-${PY_TAG}-${MAC_GLOB}.whl 2>/dev/null | head -1 || true)
    else
      DIALECT_WHL=$(ls "$WHEELHOUSE"/enigma_dialect-*-${MAC_GLOB}.whl 2>/dev/null | head -1 || true)
    fi
    DSL_WHL=$(ls "$WHEELHOUSE"/enigma_dsl-*-py3-none-any.whl 2>/dev/null | head -1 || true)
    [[ -n "$DIALECT_WHL" ]] && PKGS+=("$DIALECT_WHL")
    [[ -n "$DSL_WHL"     ]] && PKGS+=("$DSL_WHL")
  fi
  if [[ ${#PKGS[@]} -eq 0 ]]; then
    echo "    (no wheels in $WHEELHOUSE; nothing to install)"
    return
  fi
  # Run pip without the LLVM activate.sh PYTHONPATH so it doesn't shadow the venv.
  env -u PYTHONPATH "$VENV/bin/pip" install --quiet --force-reinstall --no-deps "${PKGS[@]}"
  env -u PYTHONPATH "$VENV/bin/pip" install --quiet numpy pytest >/dev/null 2>&1 || true
}

run_tests_in() {
  local VENV="$1"
  echo
  echo "==> Running pytest with $VENV"
  cd "$DSL_DIR"
  # The LLVM activate.sh prepends $ENIGMA_LLVM_INSTALL/python_packages/mlir_core
  # to PYTHONPATH, which shadows the venv's pip-installed `mlir/dialects/enigma`.
  # Strip it so pytest sees the dialect wheel we just installed.
  env -u PYTHONPATH "$VENV/bin/python" -m pytest tests/ -q
}

# --- Build phase ------------------------------------------------------------
if [[ $SKIP_DSL -eq 0 ]]; then
  build_dsl
fi

if [[ $SKIP_DIALECT -eq 0 ]]; then
  for V in "${PY_VERSIONS[@]}"; do
    for MAC in "${MACOS_TARGETS[@]}"; do
      build_dialect_for "$V" "$MAC"
    done
  done
fi

# --- Merge phase (one wheel containing dsl + dialect, per Python ABI) ------
if [[ $DO_MERGE -eq 1 ]]; then
  for V in "${PY_VERSIONS[@]}"; do
    for MAC in "${MACOS_TARGETS[@]}"; do
      merge_wheels_for "$V" "$MAC"
    done
  done
fi

# --- Install + test phase ---------------------------------------------------
# Only install/test on the current host's macOS target — wheels built for
# other targets are produced for distribution, not for local use.
_HOST_MAC_MAJ="$(sw_vers -productVersion | cut -d. -f1).0"
if [[ $DO_INSTALL -eq 1 ]]; then
  for V in "${PY_VERSIONS[@]}"; do
    VENV=$(ensure_venv_for "$V")
    install_into "$VENV" "$V" "$_HOST_MAC_MAJ"
    if [[ $DO_TEST -eq 1 ]]; then
      run_tests_in "$VENV"
    fi
  done
fi

echo
echo "============================================================"
echo "Done. Wheels available in: $WHEELHOUSE"
ls -1 "$WHEELHOUSE"
echo "============================================================"
