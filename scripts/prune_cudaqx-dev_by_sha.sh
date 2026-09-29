#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This script is intended to be run by CUDA-QX maintainers, not regular
# users.

# This script is used to prune the old cudaqx-dev packages from the GitHub
# Packages.  In general, the .github/workflows/build_dev.yaml creates new images
# whenever the CUDA-Q commit is updated, so this script will need to be
# periodically run to keep the number of packages in the GitHub Packages
# manageable.

# The script requires a TOKEN environment variable to be set by the user. The
# token must have read:packages and delete:packages scopes.

set -euo pipefail

JSON_FILE="versions.json"
ORG="NVIDIA"
PKG="cudaqx-dev"
FORCE=false
SKIP_CLEANUP=false
PRUNE_ORPHANS=false
DEBUG=false
NO_CACHE=false
CACHE_DIR="${PRUNE_CACHE_DIR:-$HOME/.cache/prune-cudaqx-dev}"

usage() {
  cat <<EOF
Usage: $0 [--force] [--skip-cleanup] [--prune-orphans] [--debug]
          [--no-cache] [--cache-dir DIR]

- Reads GitHub Packages API for $ORG/$PKG/versions
- Resolves each tagged version's manifest + OCI referrers from ghcr.io so it
  can tell which untagged versions are attestations/SBOMs/multi-arch children
  of a tagged image (REF) vs genuinely orphaned (ORPHAN).
- Default mode: prompts for an 8-char SHA, then deletes every TAGGED version
  matching that SHA *plus* its REF children.
- --prune-orphans: skips the SHA prompt and instead deletes all ORPHAN
  (untagged + unreferenced) versions.
- Blocks if any selected version carries a semver tag (x.y.z) unless --force.
- Always confirms before deleting.
- --debug: print HTTP codes for every ghcr.io call, dump each response body
  to \$TMP_DIR/debug/<id>/, and error if an OCI index manifest reports zero
  children (which would indicate a misclassification risk). Pair with
  --skip-cleanup to keep the dumps after the script exits.
- --no-cache: disable the manifest/referrers cache; always re-fetch from
  ghcr.io (useful if you suspect the cache is corrupt).
- --cache-dir DIR: use DIR as the cache directory instead of the default
  (\$PRUNE_CACHE_DIR if set, otherwise ~/.cache/prune-cudaqx-dev).
  Manifest + referrer results are keyed by version id + digest and reused on
  subsequent runs so only new/changed versions require a live fetch.

Examples:
  $0
  $0 --force
  $0 --skip-cleanup
  $0 --prune-orphans
  $0 --debug --skip-cleanup
  $0 --no-cache
  $0 --cache-dir /tmp/my-cache
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=true; shift ;;
    --skip-cleanup) SKIP_CLEANUP=true; shift ;;
    --prune-orphans) PRUNE_ORPHANS=true; shift ;;
    --debug) DEBUG=true; shift ;;
    --no-cache) NO_CACHE=true; shift ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${TOKEN:-}" ]]; then
  echo "ERROR: TOKEN is not set. You must set the TOKEN environment variable, and"
  echo "it must have read:packages and delete:packages scopes."
  exit 1
fi

TMP_DIR=$(mktemp -d)
echo "TMP_DIR: $TMP_DIR"
cd "$TMP_DIR" || exit 1

if [[ "$SKIP_CLEANUP" != "true" ]]; then
  trap 'rm -rf "$TMP_DIR"' EXIT
fi

ORG_LC=$(echo "$ORG" | tr '[:upper:]' '[:lower:]')
API_BASE="https://api.github.com/orgs/${ORG}/packages/container/${PKG}/versions"
REGISTRY_BASE="https://ghcr.io/v2/${ORG_LC}/${PKG}"

# --- 0) Exchange PAT for a short-lived ghcr.io registry token --------------- #
#
# api.github.com accepts the PAT directly, but ghcr.io's OCI distribution
# endpoint (/v2/...) does not -- it requires a registry token obtained from
# the auth realm. Without this exchange, every manifest/referrers fetch comes
# back as 403 "invalid token" and everything looks like an ORPHAN.

echo "Exchanging PAT for ghcr.io registry token..."
GHCR_TOKEN_RESP=$(curl -sS -u "ghuser:$TOKEN" \
  "https://ghcr.io/token?service=ghcr.io&scope=repository:${ORG_LC}/${PKG}:pull")
GHCR_TOKEN=$(echo "$GHCR_TOKEN_RESP" | jq -r 'try .token // empty' 2>/dev/null || true)
if [[ -z "$GHCR_TOKEN" || "$GHCR_TOKEN" == "null" ]]; then
  echo "ERROR: failed to obtain ghcr.io registry token. Response was:"
  echo "$GHCR_TOKEN_RESP"
  exit 1
fi

# --- 1) Download the version list ------------------------------------------- #

echo "Downloading $JSON_FILE from GitHub Packages..."
HTTP_CODE=$(curl -sS -o "$JSON_FILE" -w "%{http_code}" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.github+json" \
  "${API_BASE}?per_page=100")

if [[ "$HTTP_CODE" != "200" ]]; then
  echo "ERROR: GitHub API returned HTTP $HTTP_CODE"
  echo "Response body:"
  cat "$JSON_FILE"
  exit 1
fi

if ! jq -e 'type == "array"' "$JSON_FILE" > /dev/null 2>&1; then
  echo "ERROR: Expected a JSON array of versions but got:"
  cat "$JSON_FILE"
  exit 1
fi

# --- 2) Build versions.tsv:  id <TAB> digest <TAB> comma-separated tags ----- #

jq -r '
  .[]
  | [
      .id,
      (.name // ""),
      ((.metadata?.container?.tags // []) | join(","))
    ]
  | @tsv
' "$JSON_FILE" > versions.tsv

# --- 3) Resolve references for every tagged version ------------------------- #
#
# For each tagged version we fetch:
#   * its manifest        (so multi-arch index children are pulled in)
#   * its OCI referrers   (build attestations / SBOM / provenance)
# Anything either of those points at is recorded in referenced.tsv as
#   <child_digest> <TAB> <tagged_parent_id>
# so untagged versions can later be classified as REF (kept with parent) vs
# ORPHAN (safe to delete on their own).

: > referenced.tsv
DEBUG_DIR="$TMP_DIR/debug"
[[ "$DEBUG" == "true" ]] && mkdir -p "$DEBUG_DIR"

# Collected here; the run aborts after the resolution loop if non-empty.
# Anything in this list means we could not reliably tell what references what,
# so classification (and therefore --prune-orphans deletion) would be unsafe.
RESOLUTION_PROBLEMS=()

# We send Accept as multiple headers (one per type) to match how the Docker
# CLI talks to registries; some servers handle that more reliably than one
# comma-joined value when choosing between an OCI index and a v2 manifest.
manifest_accept_args=(
  -H "Accept: application/vnd.oci.image.index.v1+json"
  -H "Accept: application/vnd.docker.distribution.manifest.list.v2+json"
  -H "Accept: application/vnd.oci.image.manifest.v1+json"
  -H "Accept: application/vnd.docker.distribution.manifest.v2+json"
)

# resolve_refs_for id digest out_file
#   Fetches the manifest + OCI referrers for the given digest and writes each
#   referenced child digest (one per line) to out_file.  Resolution problems
#   are appended to the global RESOLUTION_PROBLEMS array.
resolve_refs_for() {
  local id="$1" digest="$2" out_file="$3"
  local dbg_id_dir=""
  if [[ "$DEBUG" == "true" ]]; then
    dbg_id_dir="$DEBUG_DIR/$id"
    mkdir -p "$dbg_id_dir"
  fi

  local manifest_file referrers_file
  manifest_file=$(mktemp -p "$TMP_DIR" "manifest.$id.XXXXXX.json")
  referrers_file=$(mktemp -p "$TMP_DIR" "referrers.$id.XXXXXX.json")

  # --- manifest ---
  local code child_count=0
  code=$(curl -sSL -o "$manifest_file" -w "%{http_code}" \
    -H "Authorization: Bearer $GHCR_TOKEN" \
    "${manifest_accept_args[@]}" \
    "${REGISTRY_BASE}/manifests/${digest}" || echo "000")

  if [[ "$code" == "200" ]]; then
    while IFS= read -r child; do
      [[ -z "$child" ]] && continue
      echo "$child" >> "$out_file"
      child_count=$((child_count+1))
    done < <(jq -r 'try .manifests[]?.digest // empty' "$manifest_file" 2>/dev/null)
  fi

  local mt="<none>"
  local is_index=false
  if [[ "$code" == "200" ]]; then
    mt=$(jq -r 'try .mediaType // "<none>"' "$manifest_file" 2>/dev/null || echo "<unparseable>")
    case "$mt" in
      *image.index*|*manifest.list*) is_index=true ;;
    esac
  fi

  if [[ "$DEBUG" == "true" ]]; then
    cp "$manifest_file" "$dbg_id_dir/manifest.json"
    echo "  [debug id=$id] manifest  HTTP=$code  mediaType=$mt  children=$child_count" >&2
  fi

  # --- referrers (attestations / SBOMs attached via the OCI Referrers API) ---
  local rcode ref_count=0
  rcode=$(curl -sSL -o "$referrers_file" -w "%{http_code}" \
    -H "Authorization: Bearer $GHCR_TOKEN" \
    -H "Accept: application/vnd.oci.image.index.v1+json" \
    "${REGISTRY_BASE}/referrers/${digest}" || echo "000")

  if [[ "$rcode" == "200" ]]; then
    while IFS= read -r ref; do
      [[ -z "$ref" ]] && continue
      echo "$ref" >> "$out_file"
      ref_count=$((ref_count+1))
    done < <(jq -r 'try .manifests[]?.digest // empty' "$referrers_file" 2>/dev/null)
  fi

  if [[ "$DEBUG" == "true" ]]; then
    cp "$referrers_file" "$dbg_id_dir/referrers.json"
    echo "  [debug id=$id] referrers HTTP=$rcode  referrers=$ref_count" >&2
  fi

  # Hard checks (run regardless of --debug):
  #  - manifest fetch must be 200
  #  - if the manifest IS an OCI index / docker manifest list, it must list at
  #    least one child; an empty index means we'd misclassify whatever it was
  #    supposed to reference
  #  - referrers fetch must be 200 OR 404 (404 is the documented "no referrers
  #    / not supported" response and is normal for buildx images that embed
  #    their attestation directly inside the index, or for legacy non-buildx
  #    pushes with no attestation at all)
  if [[ "$code" != "200" ]]; then
    RESOLUTION_PROBLEMS+=("id=$id digest=$digest -- manifest fetch failed (HTTP $code)")
  elif [[ "$is_index" == "true" ]] && [[ "$child_count" -eq 0 ]]; then
    RESOLUTION_PROBLEMS+=("id=$id digest=$digest -- mediaType=$mt is an index but reports 0 children")
  fi
  if [[ "$rcode" != "200" && "$rcode" != "404" ]]; then
    RESOLUTION_PROBLEMS+=("id=$id digest=$digest -- referrers fetch returned HTTP $rcode (only 200 or 404 are acceptable)")
  fi

  rm -f "$manifest_file" "$referrers_file"
}

NUM_TAGGED=$(awk -F'\t' '$3 != "" {n++} END {print n+0}' versions.tsv)

if [[ "$NO_CACHE" != "true" ]]; then
  mkdir -p "$CACHE_DIR"
  echo "Resolving manifests + referrers for $NUM_TAGGED tagged version(s) (cache: $CACHE_DIR) ..."
else
  echo "Resolving manifests + referrers for $NUM_TAGGED tagged version(s) (cache disabled) ..."
fi

# Cache format: $CACHE_DIR/<id>.cache
#   Line 1: the version's digest (cache key — digest is content-addressed and
#            never changes for a given id, but we validate it as a safety check)
#   Lines 2+: one child digest per line (manifest children + OCI referrers)
# A cached entry is used as-is; a missing or stale entry triggers a live fetch.

i=0
num_cached=0
while IFS=$'\t' read -r id digest tags; do
  [[ -z "$tags" ]] && continue
  [[ -z "$digest" ]] && continue
  i=$((i+1))
  printf '  [%d/%d] id=%s\r' "$i" "$NUM_TAGGED" "$id"

  children_file=$(mktemp -p "$TMP_DIR" "children.$id.XXXXXX")
  cache_file="$CACHE_DIR/${id}.cache"

  use_cached=false
  if [[ "$NO_CACHE" != "true" && -f "$cache_file" ]]; then
    cached_digest=$(head -1 "$cache_file" 2>/dev/null || true)
    if [[ "$cached_digest" == "$digest" ]]; then
      use_cached=true
    fi
  fi

  if [[ "$use_cached" == "true" ]]; then
    tail -n +2 "$cache_file" > "$children_file"
    num_cached=$((num_cached+1))
  else
    resolve_refs_for "$id" "$digest" "$children_file"
    if [[ "$NO_CACHE" != "true" ]]; then
      { echo "$digest"; cat "$children_file"; } > "${cache_file}.tmp" && \
        mv "${cache_file}.tmp" "$cache_file"
    fi
  fi

  while IFS= read -r child; do
    [[ -z "$child" ]] && continue
    printf '%s\t%s\n' "$child" "$id" >> referenced.tsv
  done < "$children_file"

  rm -f "$children_file"
done < versions.tsv
echo
echo "  (${num_cached}/${NUM_TAGGED} served from cache, $((NUM_TAGGED - num_cached)) fetched live)"

# --- 3b) Abort if any tagged version could not be fully resolved ------------ #

if [[ ${#RESOLUTION_PROBLEMS[@]} -gt 0 ]]; then
  echo
  echo "ERROR: ${#RESOLUTION_PROBLEMS[@]} tagged version(s) could not be reliably resolved:"
  for p in "${RESOLUTION_PROBLEMS[@]}"; do
    echo "  - $p"
  done
  echo
  echo "Refusing to continue. If we proceeded, REF children of these tagged"
  echo "versions would be misclassified as ORPHAN, which means a --prune-orphans"
  echo "run could delete still-referenced platform manifests or attestations."
  echo
  echo "Re-run with --debug --skip-cleanup and inspect the failing responses at:"
  echo "  \$TMP_DIR/debug/<id>/manifest.json"
  echo "  \$TMP_DIR/debug/<id>/referrers.json"
  exit 1
fi

# --- 4) Pretty listing ------------------------------------------------------ #

echo
echo "Current versions:"
# Build the raw classification as TSV first; we use it both for the
# (column-formatted) display and for the summary counts. We cannot count
# from the formatted output because `column -t` strips the tabs.
{
  printf 'STATUS\tID\tDIGEST_SHORT\tTAGS_OR_PARENT\n'
  while IFS=$'\t' read -r id digest tags; do
    short="${digest#sha256:}"; short="${short:0:12}"
    if [[ -n "$tags" ]]; then
      printf 'TAGGED\t%s\t%s\t%s\n' "$id" "$short" "$tags"
    else
      parents=$(awk -F'\t' -v d="$digest" '$1==d {print $2}' referenced.tsv \
                | sort -u | paste -sd, -)
      if [[ -n "$parents" ]]; then
        printf 'REF\t%s\t%s\tparent_ids=%s\n' "$id" "$short" "$parents"
      else
        printf 'ORPHAN\t%s\t%s\t-\n' "$id" "$short"
      fi
    fi
  done < versions.tsv
} > listing.tsv
column -t -s$'\t' < listing.tsv
echo

NUM_TAGGED_TOTAL=$(awk -F'\t' 'NR>1 && $1=="TAGGED" {n++} END {print n+0}' listing.tsv)
NUM_REF_TOTAL=$(awk -F'\t'    'NR>1 && $1=="REF"    {n++} END {print n+0}' listing.tsv)
NUM_ORPHAN_TOTAL=$(awk -F'\t' 'NR>1 && $1=="ORPHAN" {n++} END {print n+0}' listing.tsv)
echo "Summary: TAGGED=$NUM_TAGGED_TOTAL  REF=$NUM_REF_TOTAL  ORPHAN=$NUM_ORPHAN_TOTAL"
echo

# --- 5) Build the deletion set --------------------------------------------- #

: > ids_to_delete.txt   # id <TAB> "short description"

if [[ "$PRUNE_ORPHANS" == "true" ]]; then
  echo "Mode: --prune-orphans"
  while IFS=$'\t' read -r id digest tags; do
    [[ -n "$tags" ]] && continue
    parents=$(awk -F'\t' -v d="$digest" '$1==d {print $2}' referenced.tsv | head -n1)
    [[ -n "$parents" ]] && continue
    short="${digest#sha256:}"; short="${short:0:12}"
    printf '%s\torphan digest=%s\n' "$id" "$short" >> ids_to_delete.txt
  done < versions.tsv
else
  read -rp "Enter the 8-char SHA to delete (e.g., 5fad63e4): " SHORTSHA
  if [[ ! "$SHORTSHA" =~ ^[0-9a-f]{8}$ ]]; then
    echo "ERROR: '$SHORTSHA' is not an 8-char lowercase hex SHA."
    exit 1
  fi

  # Tagged versions whose tag string contains the SHA.
  matching_tagged_ids=()
  while IFS=$'\t' read -r id digest tags; do
    [[ -z "$tags" ]] && continue
    if [[ ",${tags}," == *",${SHORTSHA},"* || ",${tags}," == *",${SHORTSHA}-"* ]]; then
      matching_tagged_ids+=("$id")
      short="${digest#sha256:}"; short="${short:0:12}"
      printf '%s\ttagged: %s\n' "$id" "$tags" >> ids_to_delete.txt
    fi
  done < versions.tsv

  if [[ ${#matching_tagged_ids[@]} -eq 0 ]]; then
    echo "No tagged versions matched SHA '$SHORTSHA'. Nothing to do."
    exit 0
  fi

  # Pull in untagged versions referenced by any matched tagged version.
  for parent_id in "${matching_tagged_ids[@]}"; do
    while IFS=$'\t' read -r child_digest pid; do
      [[ "$pid" != "$parent_id" ]] && continue
      child_id=$(awk -F'\t' -v d="$child_digest" '$2==d {print $1; exit}' versions.tsv)
      [[ -z "$child_id" ]] && continue
      # Skip if this child is itself tagged (shouldn't happen, but be safe).
      child_tags=$(awk -F'\t' -v i="$child_id" '$1==i {print $3; exit}' versions.tsv)
      [[ -n "$child_tags" ]] && continue
      # De-dupe.
      grep -q "^${child_id}	" ids_to_delete.txt && continue
      printf '%s\tref-child of %s (digest=%s)\n' "$child_id" "$parent_id" "${child_digest#sha256:}" >> ids_to_delete.txt
    done < referenced.tsv
  done
fi

if [[ ! -s ids_to_delete.txt ]]; then
  echo "Nothing to delete."
  exit 0
fi

echo
echo "Planned deletions:"
column -t -s$'\t' ids_to_delete.txt
echo

# --- 6) Semver guardrail ---------------------------------------------------- #

# Re-check: do any *tagged* members of the deletion set carry a semver tag?
HAS_SEMVER=false
while IFS=$'\t' read -r del_id _; do
  tags=$(awk -F'\t' -v i="$del_id" '$1==i {print $3; exit}' versions.tsv)
  [[ -z "$tags" ]] && continue
  if echo "$tags" | grep -Eq '(^|,)[0-9]+\.[0-9]+\.[0-9]+(-|,|$)'; then
    HAS_SEMVER=true
    break
  fi
done < ids_to_delete.txt

if [[ "$HAS_SEMVER" == "true" && "$FORCE" != "true" ]]; then
  echo "ABORT: One or more matching versions include a semantic version tag (e.g., 0.4.0)."
  echo "       Re-run with --force if you are REALLY sure you want to delete them."
  exit 1
fi

# --- 7) Confirm + delete ---------------------------------------------------- #

NUM=$(wc -l < ids_to_delete.txt | tr -d ' ')
echo "About to delete $NUM version(s) from org '$ORG' package '$PKG'."
read -rp "Type 'delete' to proceed: " CONFIRM
if [[ "$CONFIRM" != "delete" ]]; then
  echo "Canceled."
  exit 0
fi

while IFS=$'\t' read -r id desc; do
  if [[ ! "$id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ID '$id' is not numeric. Exiting just to be safe."
    exit 1
  fi
  if [[ ${#id} -ne 9 ]]; then
    echo "ERROR: ID '$id' is not 9 digits. Exiting just to be safe."
    exit 1
  fi
  echo "Deleting version id $id  ($desc) ..."
  resp_file=$(mktemp -p "$TMP_DIR" "delete.$id.XXXXXX.json")
  http_code=$(curl -sS -X DELETE -o "$resp_file" -w "%{http_code}" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "${API_BASE}/${id}" || echo "000")

  # GitHub returns 204 No Content for a successful package-version delete.
  # Anything else (including a body with a .message field) is a failure --
  # bail immediately so we don't keep mutating state in a broken state.
  if [[ "$http_code" == "204" || "$http_code" == "200" ]]; then
    echo "  -> deleted (HTTP $http_code)"
    rm -f "$resp_file"
  else
    msg=$(jq -r 'try .message // empty' "$resp_file" 2>/dev/null || echo "")
    echo "ERROR: DELETE failed for id $id (HTTP $http_code)"
    [[ -n "$msg" ]] && echo "       message: $msg"
    echo "       full response body:"
    sed 's/^/         /' "$resp_file" | head -n 20
    echo
    echo "Aborting before any further deletions. Re-run with --skip-cleanup to"
    echo "preserve the raw response at: $resp_file"
    exit 1
  fi
done < ids_to_delete.txt

echo "Done."
