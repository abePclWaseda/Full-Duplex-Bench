import argparse
import shutil
from pathlib import Path

import transformers

def parse_args():
    parser = argparse.ArgumentParser(
        description="Patch the transformers Whisper `generation_whisper.py` with your local copy."
    )
    parser.add_argument(
        "--src_file",
        type=Path,
        default=Path("generation_whisper.py"),
        help="Path to your modified `generation_whisper.py`",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    src = args.src_file.expanduser().resolve()
    if not src.is_file():
        print(f"Source file not found: {src}")
        return

    # Locate installed transformers package
    pkg_root = Path(transformers.__file__).parent
    target = pkg_root / "models" / "whisper" / "generation_whisper.py"

    if not target.exists():
        print(f"Could not find target file at: {target}")
        return

    # Overwrite
    shutil.copy2(src, target)
    print(f"Patched `{target}` with `{src}`")

if __name__ == "__main__":
    main()
