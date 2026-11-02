#!/bin/sh
# Compile a Stan model into a BridgeStan model library.
#
#   build-model.sh MODEL.stan OUTDIR
#
# Uses BridgeStan at $BRIDGESTAN. Without it, the script fetches BridgeStan
# into .bridgestan/ at the repository root, never into a global location.
# Needs a C++17 compiler and make.
set -eu

version=2.9.0
model=$1
outdir=$2
root=$(cd "$(dirname "$0")/../.." && pwd)

if [ -z "${BRIDGESTAN:-}" ]; then
  BRIDGESTAN="$root/.bridgestan/bridgestan-$version"
  if [ ! -f "$BRIDGESTAN/Makefile" ]; then
    mkdir -p "$root/.bridgestan"
    curl -sSfL "https://github.com/roualdes/bridgestan/releases/download/v$version/bridgestan-$version.tar.gz" \
      | tar -xz -C "$root/.bridgestan"
  fi
fi

name=$(basename "$model" .stan)
mkdir -p "$outdir"
outdir=$(cd "$outdir" && pwd)
cmp -s "$model" "$outdir/$name.stan" || cp "$model" "$outdir/$name.stan"
make -s -C "$BRIDGESTAN" -j"${JOBS:-4}" "$outdir/${name}_model.so"
echo "$outdir/${name}_model.so"
