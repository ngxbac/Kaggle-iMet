import cv2
import glob
import os
import argparse
from tqdm import *
from joblib.parallel import Parallel, delayed


def resize_and_save(image_path, dest, image_size):
    image_name = image_path.split("/")[-1]
    out_file = os.path.join(dest, image_name)

    # Resize
    image = cv2.imread(image_path)
    image = cv2.resize(image, dsize=(int(image_size), int(image_size)))
    cv2.imwrite(out_file, image)


def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    all_images = glob.glob(f"{args.input_folder}/*.png")
    Parallel(n_jobs=4)(delayed(resize_and_save)(image_path, args.output_folder, args.image_size)
                       for image_path in tqdm(all_images))


def parse_args():
    description = 'Resize image'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--image_size', dest='image_size',
                        help='Size of image',
                        default=224, type=int)

    parser.add_argument('--input_folder', dest='input_folder',
                        help='Size of image',
                        default=None, type=str)

    parser.add_argument('--output_folder', dest='output_folder',
                        help='Size of image',
                        default=None, type=str)

    return parser.parse_args()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    print('Resize image')
    args = parse_args()
    main(args)

