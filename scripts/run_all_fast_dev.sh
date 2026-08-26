#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}

"$PYTHON_BIN" scripts/prepare_oasis_data.py
"$PYTHON_BIN" scripts/smoke_tests.py
"$PYTHON_BIN" scripts/visualize_oasis_sample.py
"$PYTHON_BIN" scripts/part3_cifar_resnet18.py --fast-dev --synthetic
"$PYTHON_BIN" scripts/part1_dft.py --fast-dev
"$PYTHON_BIN" scripts/part4_vae.py --data /Users/xct/Downloads/keras_png_slices_data.zip --fast-dev
"$PYTHON_BIN" scripts/part4_unet.py --data /Users/xct/Downloads/keras_png_slices_data.zip --fast-dev
"$PYTHON_BIN" scripts/part4_gan.py --data /Users/xct/Downloads/keras_png_slices_data.zip --fast-dev
