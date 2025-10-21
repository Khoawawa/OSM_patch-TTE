import os
import argparse
from PIL import Image
from tqdm import tqdm

def resize_images(input_folder, size=(224, 224)):
    """
    Resize all images in the input folder to the given size.
    Overwrites the originals.
    """
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    for filename in tqdm(os.listdir(input_folder), desc="Resizing images"):
        if not filename.lower().endswith(supported_ext):
            continue

        in_path = os.path.join(input_folder, filename)
        out_path = os.path.join(f"{input_folder}_{size[0]}x{size[1]}", filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            with Image.open(in_path) as img:
                img = img.resize(size, Image.Resampling.LANCZOS)
                img.save(out_path)
        except Exception as e:
            print(f"⚠️ Failed: {filename} ({e})")

def main():
    parser = argparse.ArgumentParser(description="Resize all images in a folder (overwrite in place).")
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