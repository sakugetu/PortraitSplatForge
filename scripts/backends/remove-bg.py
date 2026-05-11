import argparse
import io
from pathlib import Path

from PIL import Image
from rembg import new_session, remove


def main():
    parser = argparse.ArgumentParser(description="Create a foreground alpha mask with rembg.")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("mask", help="Output grayscale mask path")
    parser.add_argument("--model", default="u2net_human_seg", help="rembg model name")
    parser.add_argument("--no-post-process", action="store_true", help="Disable rembg mask post-processing")
    args = parser.parse_args()

    input_path = Path(args.input)
    mask_path = Path(args.mask)
    session = new_session(args.model)
    with input_path.open("rb") as handle:
        mask_bytes = remove(
            handle.read(),
            session=session,
            only_mask=True,
            post_process_mask=not args.no_post_process,
            alpha_matting=False,
        )
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(mask_bytes, Image.Image):
        mask = mask_bytes
    else:
        mask = Image.open(io.BytesIO(mask_bytes))
    mask.convert("L").save(mask_path)


if __name__ == "__main__":
    main()
