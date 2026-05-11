import sys

import pycolmap


if len(sys.argv) != 2:
    raise SystemExit("usage: write-colmap-text.py <sparse-dir>")

reconstruction = pycolmap.Reconstruction(sys.argv[1])
reconstruction.write_text(sys.argv[1])
