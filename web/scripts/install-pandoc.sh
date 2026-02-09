#!/bin/bash
# Download pandoc static binary for CI (Linux amd64)
# Skips if pandoc is already available

set -e

if command -v pandoc &> /dev/null; then
  echo "pandoc already installed: $(pandoc --version | head -1)"
  exit 0
fi

if [ -f ./bin/pandoc ]; then
  echo "pandoc binary already downloaded"
  exit 0
fi

PANDOC_VERSION="3.6.2"
PANDOC_URL="https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/pandoc-${PANDOC_VERSION}-linux-amd64.tar.gz"

echo "Downloading pandoc ${PANDOC_VERSION}..."
mkdir -p ./bin
curl -sL "$PANDOC_URL" | tar xz --strip-components=2 -C ./bin "pandoc-${PANDOC_VERSION}/bin/pandoc"
chmod +x ./bin/pandoc
echo "pandoc installed to ./bin/pandoc"
./bin/pandoc --version | head -1
