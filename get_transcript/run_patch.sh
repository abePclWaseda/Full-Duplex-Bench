set -euo pipefail

echo "Installing transformers from GitHub (branch: crisper_whisper)…"
pip install --no-cache-dir git+https://github.com/nyrahealth/transformers.git@crisper_whisper

echo "Running patch_transformers.py…"
python patch_transformers.py

echo "Done."
