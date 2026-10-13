#!/bin/bash
# Clone GenD (MIT licence) at the commit used for every result in the paper.
set -e
cd "$(dirname "$0")"
mkdir -p third_party
[ -d third_party/GenD ] || git clone https://github.com/yermandy/GenD.git third_party/GenD
git -C third_party/GenD checkout 387a42266dd385fbe8f7626c5d3aa03eea3bfaab
echo "GenD ready in third_party/GenD; install its requirements with: pip install -r third_party/GenD/requirements.txt"
