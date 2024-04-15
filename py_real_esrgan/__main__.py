import argparse
from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image

from py_real_esrgan.model import RealESRGAN


def main():
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model-directory", type=Path, default=Path.cwd() / "model", help="Model directory")
    parser.add_argument("-s", "--scale", type=int, default=2, help="Scale factor")
    parser.add_argument("-i", "--input-directory", type=Path, default=Path.cwd() / "inputs", help="Input directory")
    parser.add_argument("-o", "--output-directory", type=Path, default=Path.cwd() / "outputs", help="Output directory")
    args = parser.parse_args()

    if args.scale not in [2, 4, 8]:
        print("Scale factor must be 2 or 4 or 8")
        return

    if not args.input_directory.exists():
        print(f"Input directory '{args.input_directory}' does not exist")
        return

    if not args.model_directory.exists():
        args.model_directory.mkdir(exist_ok=True)

    if not args.output_directory.exists():
        args.output_directory.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=args.scale)
    model.load_weights(f'{args.model_directory}/RealESRGAN_x{args.scale}.pth', download=True)

    for path in args.input_directory.iterdir():
        print(f"Processing {path}")
        image = Image.open(path).convert('RGB')
        sr_image = model.predict(image)
        sr_image.save(args.output_directory / path.name)

    print("Done")


if __name__ == '__main__':
    main()
