import os
import json
import argparse
from PIL import Image
from tqdm import tqdm

def resize_images(input_folder, size=(224, 224)):
    """
    Resize all images in the input folder to the given size and
    save to a new folder named <input_folder>_<WxH>.
    Also updates metadata JSON if found.
    """
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    out_folder = f"{input_folder}_{size[0]}x{size[1]}"
    os.makedirs(out_folder, exist_ok=True)

    # Resize all images
    for filename in tqdm(os.listdir(input_folder), desc="Resizing images"):
        if not filename.lower().endswith(supported_ext):
            continue

        in_path = os.path.join(input_folder, filename)
        out_path = os.path.join(out_folder, filename)

        try:
            with Image.open(in_path) as img:
                img = img.resize(size, Image.Resampling.LANCZOS)
                img.save(out_path)
        except Exception as e:
            print(f"⚠️ Failed: {filename} ({e})")

    # Update metadata if exists
    meta_path = os.path.join(input_folder, "patch_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            for item in metadata:
                if "image_path" in item:
                    old_path = item["image_path"]
                    filename = os.path.basename(old_path)
                    item["image_path"] = os.path.join(out_folder, filename).replace("\\", "/")

            out_meta = os.path.join(out_folder, "patch_metadata.json")
            with open(out_meta, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to update metadata: {e}")
    else:
        print("ℹ️ No metadata file found, skipping update.")

def main():
    parser = argparse.ArgumentParser(description="Resize all images in a folder and update metadata.")
    parser.add_argument("-i", "--input", help="Input folder containing images.", default="./mydata/patches")
    parser.add_argument("-s", "--size", type=int, nargs='+', required=True,
                        help="Target size (one value for square resize, or two for width height).")

    args = parser.parse_args()

    if len(args.size) == 1:
        size = (args.size[0], args.size[0])
    elif len(args.size) == 2:
        size = (args.size[0], args.size[1])
    else:
        raise ValueError("Invalid size argument. Use one (e.g. -s 112) or two values (e.g. -s 128 96).")

    resize_images(args.input, size=size)

if __name__ == "__main__":
    main()
