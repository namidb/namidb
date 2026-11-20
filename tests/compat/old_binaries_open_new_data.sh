#!/usr/bin/env bash
# Rollback check: data written by this tree must open, serve correct results,
# take writes, flush and compact under the previous releases, and roll
# forward again.
#
#   tests/compat/old_binaries_open_new_data.sh [new-server-binary]
#
# The old servers are built from their release tags (vector-index,text-index)
# into $NAMIDB_COMPAT_CACHE (default: target/compat) on first use. Set
# NAMIDB_COMPAT_TAGS to change the set (default: "v3.4.3 v3.5.0 v3.5.1"), or
# NAMIDB_COMPAT_BIN_<tag with dots as underscores> (e.g.
# NAMIDB_COMPAT_BIN_v3_5_1) to use an existing binary built from that tag.
set -euo pipefail
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cache=${NAMIDB_COMPAT_CACHE:-$repo/target/compat}
tags=${NAMIDB_COMPAT_TAGS:-"v3.4.3 v3.5.0 v3.5.1"}
new=${1:-}
if [ -z "$new" ]; then
  (cd "$repo" && cargo build --locked --release -p namidb-server --features vector-index,text-index)
  new=$repo/target/release/namidb-server
fi
args=()
for tag in $tags; do
  var="NAMIDB_COMPAT_BIN_${tag//./_}"
  bin=${!var:-$cache/$tag/namidb-server}
  if [ ! -x "$bin" ]; then
    src=$cache/src-$tag
    rm -rf "$src"
    git -C "$repo" worktree add --detach "$src" "$tag" >/dev/null
    (cd "$src" && CARGO_TARGET_DIR=$cache/target cargo build --locked --release \
      -p namidb-server --features vector-index,text-index)
    mkdir -p "$cache/$tag"
    cp "$cache/target/release/namidb-server" "$cache/$tag/namidb-server"
    git -C "$repo" worktree remove --force "$src"
    bin=$cache/$tag/namidb-server
  fi
  args+=(--old "$tag=$bin")
done
exec python3 "$repo/tests/compat/old_binaries_open_new_data.py" --new "$new" "${args[@]}"
