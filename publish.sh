#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
GITHUB_ORG="langguard-ai"
IMAGE_NAME="opencite"
REGISTRY="ghcr.io"
FULL_IMAGE="${REGISTRY}/${GITHUB_ORG}/${IMAGE_NAME}"

# ─── Helpers ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Publish the Open-CITE Docker image to GitHub Container Registry.

Options:
  --version VERSION   Set version (required — prompted if not provided)
  --skip-login        Skip the GHCR login step
  --skip-release      Skip the GitHub release step
  --multi-arch        Build for linux/amd64 and linux/arm64
  --dry-run           Show what would be done without executing
  -h, --help          Show this help message

Examples:
  ./publish.sh                      # Prompted for version
  ./publish.sh --version 0.2.0      # Set version explicitly
  ./publish.sh --multi-arch         # Multi-arch build (amd64 + arm64)
  ./publish.sh --dry-run            # Preview commands without running them
EOF
    exit 0
}

run() {
    if $DRY_RUN; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $*"
    else
        "$@"
    fi
}

# ─── Parse arguments ────────────────────────────────────────────────────────
VERSION=""
SKIP_LOGIN=false
SKIP_RELEASE=false
MULTI_ARCH=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)      VERSION="$2"; shift 2 ;;
        --skip-login)   SKIP_LOGIN=true; shift ;;
        --skip-release) SKIP_RELEASE=true; shift ;;
        --multi-arch)   MULTI_ARCH=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        -h|--help)      usage ;;
        *)              error "Unknown option: $1. Use --help for usage." ;;
    esac
done

# ─── Gather all inputs up front ───────────────────────────────────────────

# Fetch the latest tagged version and suggest next patch bump
git fetch --tags --quiet 2>/dev/null || true
LATEST_TAG=$(git tag --sort=-v:refname --list 'v[0-9]*' | head -1)
if [[ -n "$LATEST_TAG" ]]; then
    IFS='.' read -r MAJOR MINOR PATCH <<< "${LATEST_TAG#v}"
    SUGGESTED_VERSION="${MAJOR}.${MINOR}.$(( PATCH + 1 ))"
    info "Latest tag:       ${LATEST_TAG}"
    info "Suggested version: ${SUGGESTED_VERSION}"
else
    SUGGESTED_VERSION="0.1.0"
    info "No existing tags found."
    info "Suggested version: ${SUGGESTED_VERSION}"
fi
echo

# Prompt for version if not provided via --version
if [[ -z "$VERSION" ]]; then
    read -rp "Version [${SUGGESTED_VERSION}]: " VERSION
    VERSION="${VERSION:-${SUGGESTED_VERSION}}"
fi

# Validate version is not empty
[[ -z "$VERSION" ]] && error "Version is required."

TAG="v${VERSION}"

# Prompt for release inputs if not skipping
RELEASE_TITLE=""
RELEASE_BODY=""
if ! $SKIP_RELEASE; then
    if ! command -v gh &>/dev/null; then
        warn "gh CLI not installed — will skip GitHub release."
        SKIP_RELEASE=true
    else
        read -rp "Release title [${TAG}]: " RELEASE_TITLE
        RELEASE_TITLE="${RELEASE_TITLE:-${TAG}}"

        echo "Release description (enter an empty line to finish, or leave blank for auto-generated notes):"
        while IFS= read -r line; do
            [[ -z "$line" ]] && break
            RELEASE_BODY="${RELEASE_BODY}${line}"$'\n'
        done
    fi
fi

# ─── Summary ──────────────────────────────────────────────────────────────
echo
info "Image:   ${FULL_IMAGE}"
info "Version: ${VERSION}"
info "Tags:    ${VERSION}, latest"
info "Release: ${TAG}"
echo

# ─── Preflight checks ────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    error "Docker is not installed or not in PATH."
fi

if $MULTI_ARCH; then
    if ! docker buildx version &>/dev/null; then
        error "docker buildx is required for multi-arch builds. Install it first."
    fi
fi

# ─── Login to GHCR ───────────────────────────────────────────────────────
if ! $SKIP_LOGIN; then
    info "Logging in to ${REGISTRY}..."
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        echo "${GITHUB_TOKEN}" | run docker login "${REGISTRY}" -u "${GITHUB_ORG}" --password-stdin
    else
        warn "GITHUB_TOKEN not set. Attempting interactive login..."
        warn "Create a PAT at: https://github.com/settings/tokens/new"
        warn "Required scope: write:packages"
        echo
        run docker login "${REGISTRY}"
    fi
    info "Login successful."
else
    info "Skipping login (--skip-login)."
fi

# ─── Build and push ──────────────────────────────────────────────────────
if $MULTI_ARCH; then
    info "Building multi-arch image (linux/amd64, linux/arm64)..."

    # Ensure a buildx builder exists
    if ! docker buildx inspect opencite-builder &>/dev/null; then
        info "Creating buildx builder..."
        run docker buildx create --name opencite-builder --use
    else
        run docker buildx use opencite-builder
    fi

    run docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --tag "${FULL_IMAGE}:${VERSION}" \
        --tag "${FULL_IMAGE}:latest" \
        --label "org.opencontainers.image.source=https://github.com/LangGuard-AI/opencite" \
        --label "org.opencontainers.image.version=${VERSION}" \
        --label "org.opencontainers.image.description=Open-CITE - Open-source AI asset discovery" \
        --push \
        .

else
    info "Building image..."
    run docker build \
        --tag "${FULL_IMAGE}:${VERSION}" \
        --tag "${FULL_IMAGE}:latest" \
        --label "org.opencontainers.image.source=https://github.com/LangGuard-AI/opencite" \
        --label "org.opencontainers.image.version=${VERSION}" \
        --label "org.opencontainers.image.description=Open-CITE - Open-source AI asset discovery" \
        .

    info "Pushing ${FULL_IMAGE}:${VERSION}..."
    run docker push "${FULL_IMAGE}:${VERSION}"

    info "Pushing ${FULL_IMAGE}:latest..."
    run docker push "${FULL_IMAGE}:latest"
fi

# ─── GitHub Release ───────────────────────────────────────────────────────
if ! $SKIP_RELEASE; then
    echo
    if $DRY_RUN; then
        echo -e "${YELLOW}[DRY-RUN]${NC} gh release create ${TAG} --title \"${RELEASE_TITLE}\" --notes \"...\""
    else
        if [[ -n "$RELEASE_BODY" ]]; then
            gh release create "${TAG}" \
                --title "${RELEASE_TITLE}" \
                --notes "${RELEASE_BODY}"
        else
            gh release create "${TAG}" \
                --title "${RELEASE_TITLE}" \
                --generate-notes
        fi
        info "GitHub release ${TAG} created."
    fi
else
    info "Skipping GitHub release (--skip-release)."
fi

# ─── Done ─────────────────────────────────────────────────────────────────
echo
info "Published successfully!"
info "Pull with: docker pull ${FULL_IMAGE}:${VERSION}"
