#!/bin/bash
# Detect and align the face in every photo in my_images/, score it with the three detectors,
# and save the table and the image grid in results_my_images/.
cd "$(dirname "$0")"
/venv/gend/bin/python demo.py --images my_images --detect --out results_my_images "$@"
