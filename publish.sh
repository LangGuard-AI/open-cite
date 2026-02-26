#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
GITHUB_ORG="langguard-ai"
IMAGE_NAME="opencite"
REGISTRY="ghcr.io"
FULL_IMAGE="${REGISTRY}/${GITHUB_ORG}/${IMAGE_NAME}"

# Read version from pyproject.toml
VERSION=$(grep -m1 'version' pyproject.toml | sed 's/.*"\(.*\)".*/\1/')

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
  --version VERSION   Override version tag (default: ${VERSION} from pyproject.toml)
  --skip-login        Skip the GHCR login step
  --multi-arch        Build for linux/amd64 and linux/arm64
  --dry-run           Show what would be done without executing
  -h, --help          Show this help message

Examples:
  ./publish.sh                      # Build and push v${VERSION}
  ./publish.sh --version 0.2.0      # Override version
  ./publish.sh --multi-arch         # Multi-arch build (amd64 + arm64)
  ./publish.sh --dry-run            # Preview commands without running them
EOF
    exit 0
}

# ─── Parse arguments ────────────────────────────────────────────────────────
SKIP_LOGIN=false
MULTI_ARCH=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)    VERSION="$2"; shift 2 ;;
        --skip-login) SKIP_LOGIN=true; shift ;;
        --multi-arch) MULTI_ARCH=true; shift ;;
        --dry-run)    DRY_RUN=true; shift ;;
        -h|--help)    usage ;;
        *)            error "Unknown option: $1. Use --help for usage." ;;
    esac
done

run() {
    if $DRY_RUN; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $*"
    else
        "$@"
    fi
}

# ─── Preflight checks ──────────────────────────────────────────────────────
info "Image:   ${FULL_IMAGE}"
info "Version: ${VERSION}"
info "Tags:    ${VERSION}, latest"
echo

if ! command -v docker &>/dev/null; then
    error "Docker is not installed or not in PATH."
fi

if $MULTI_ARCH; then
    if ! docker buildx version &>/dev/null; then
        error "docker buildx is required for multi-arch builds. Install it first."
    fi
fi

# ─── Login to GHCR ─────────────────────────────────────────────────────────
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

# ─── Build and push ────────────────────────────────────────────────────────
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

# ─── Done ───────────────────────────────────────────────────────────────────
echo
info "Published successfully!"
info "Pull with: docker pull ${FULL_IMAGE}:${VERSION}"
