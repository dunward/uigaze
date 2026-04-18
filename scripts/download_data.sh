#!/usr/bin/env bash
#
# UEyes dataset download script
# Source: https://zenodo.org/record/8010312 (CHI 2023, Yue Jiang et al.)
# License: CC BY 4.0
#
# Usage:
#   chmod +x scripts/download_data.sh
#   ./scripts/download_data.sh
#
# Downloads ~12.9GB. Run on a stable network connection.

set -euo pipefail

DATA_DIR="data"
ZIP_FILE="${DATA_DIR}/UEyes_dataset.zip"
DOWNLOAD_URL="https://zenodo.org/api/records/8010312/files/UEyes_dataset.zip/content"
EXPECTED_MD5="c2d53e6af0a47e1f459416d6839ec2c1"

echo "============================================"
echo "UIGaze — UEyes Dataset Downloader"
echo "============================================"
echo ""

# Create data directory
mkdir -p "${DATA_DIR}"

# Check if already extracted
if [ -f "${DATA_DIR}/info.csv" ] && [ -d "${DATA_DIR}/images" ]; then
    echo "Dataset already exists in ${DATA_DIR}/"
    echo "To re-download, remove the data/ directory first."
    exit 0
fi

# Download
if [ -f "${ZIP_FILE}" ]; then
    echo "Zip file already exists, skipping download."
else
    echo "Downloading UEyes dataset from Zenodo (~12.9GB)..."
    echo "URL: ${DOWNLOAD_URL}"
    echo ""

    if command -v curl &> /dev/null; then
        curl -L --progress-bar -o "${ZIP_FILE}" "${DOWNLOAD_URL}"
    elif command -v wget &> /dev/null; then
        wget --show-progress -O "${ZIP_FILE}" "${DOWNLOAD_URL}"
    else
        echo "ERROR: curl or wget is required."
        exit 1
    fi

    echo ""
    echo "Download complete."
fi

# Verify checksum
echo "Verifying checksum..."
if command -v md5sum &> /dev/null; then
    ACTUAL_MD5=$(md5sum "${ZIP_FILE}" | awk '{print $1}')
elif command -v md5 &> /dev/null; then
    ACTUAL_MD5=$(md5 -q "${ZIP_FILE}")
else
    echo "WARNING: md5sum/md5 not found, skipping checksum verification."
    ACTUAL_MD5="${EXPECTED_MD5}"
fi

if [ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]; then
    echo "ERROR: Checksum mismatch!"
    echo "  Expected: ${EXPECTED_MD5}"
    echo "  Actual:   ${ACTUAL_MD5}"
    echo "The file may be corrupted. Delete ${ZIP_FILE} and try again."
    exit 1
fi
echo "Checksum OK."

# Extract
echo ""
echo "Extracting dataset..."
unzip -q -o "${ZIP_FILE}" -d "${DATA_DIR}"

# The zip may contain a top-level directory (e.g., UEyes_dataset/).
# Move contents up if needed so data/images/ exists directly.
if [ ! -d "${DATA_DIR}/images" ]; then
    # Find the extracted subdirectory
    SUBDIR=$(find "${DATA_DIR}" -maxdepth 1 -mindepth 1 -type d | head -1)
    if [ -n "${SUBDIR}" ] && [ -d "${SUBDIR}/images" ]; then
        echo "Moving contents from ${SUBDIR}/ to ${DATA_DIR}/..."
        mv "${SUBDIR}"/* "${DATA_DIR}/"
        rmdir "${SUBDIR}"
    else
        echo "WARNING: Unexpected zip structure. Please check ${DATA_DIR}/ manually."
    fi
fi

# Clean up zip
echo ""
read -p "Delete zip file to save disk space? (y/N): " CONFIRM
if [[ "${CONFIRM}" =~ ^[Yy]$ ]]; then
    rm -f "${ZIP_FILE}"
    echo "Zip file deleted."
else
    echo "Zip file kept at ${ZIP_FILE}"
fi

# Verify structure
echo ""
echo "============================================"
echo "Verifying dataset structure..."
echo "============================================"

CHECKS_PASSED=true

for DIR in "images" "saliency_maps/heatmaps/3s" "saliency_maps/fixmaps/3s" "eyetracker_logs"; do
    if [ -d "${DATA_DIR}/${DIR}" ]; then
        COUNT=$(find "${DATA_DIR}/${DIR}" -maxdepth 1 -type f | wc -l | tr -d ' ')
        echo "  ✓ ${DIR}/ (${COUNT} files)"
    else
        echo "  ✗ ${DIR}/ — NOT FOUND"
        CHECKS_PASSED=false
    fi
done

if [ -f "${DATA_DIR}/info.csv" ]; then
    LINES=$(wc -l < "${DATA_DIR}/info.csv" | tr -d ' ')
    echo "  ✓ info.csv (${LINES} lines)"
else
    echo "  ✗ info.csv — NOT FOUND"
    CHECKS_PASSED=false
fi

echo ""
if [ "${CHECKS_PASSED}" = true ]; then
    echo "Dataset ready! Run the pilot experiment:"
    echo "  uv run python experiments/run_pilot.py"
else
    echo "WARNING: Some expected files/folders are missing."
    echo "The zip structure may differ. Check data/ and adjust paths if needed."
fi
