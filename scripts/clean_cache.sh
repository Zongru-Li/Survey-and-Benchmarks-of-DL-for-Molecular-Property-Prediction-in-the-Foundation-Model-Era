#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "Cleaning .pth files from tmp/..."
find tmp -name "*.pth" -type f -delete -print | wc -l | xargs echo "Deleted"
echo "Done."
