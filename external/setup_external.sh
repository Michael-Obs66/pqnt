#!/bin/sh
set -e

echo "[pqnt] Setting up external dependencies..."

# Create external folder if needed
mkdir -p external

# Check if pybind11 already exists
if [ -d external/pybind11 ]; then
    echo "[pqnt] pybind11 already exists. Skipping download."
else
    echo "[pqnt] Downloading pybind11 into external/ ..."
    git clone https://github.com/pybind/pybind11.git external/pybind11
fi

echo "[pqnt] External dependencies ready."
