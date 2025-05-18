import numpy as np
from PIL import Image
import argparse
from models import apply_matrix
from find_solution import get_matrix
from dataset_processing import create_train_dataset, srgb_to_linear, linear_to_srgb



def parse_args():
    parser = argparse.ArgumentParser("Main Benchmark Utility")

    parser.add_argument(
        "-o", 
        "--output_dir", 
        required=True, 
        help="directiry to save resulting images"
    )

    parser.add_argument(
        "-i",
        "--index",
        required=True,
        type=int,
        nargs='+',
        help="number of piture in dataset (from 1 to 197)"
    )

    parser.add_argument(
        "-m",
        "--models",
        nargs='+',
        help="model to use for training",
        default=["PCC2", "PCC3", "RPCC2", "RPCC3", "linear"],
        choices=["PCC2", "PCC3", "RPCC2", "RPCC3", "linear"],
    )

    parser.add_argument(
        "--camera_1",
        default='Canon1DsMkIII',
        help="name of first camera"
    )

    parser.add_argument(
        "--camera_2",
        default='Canon600D',
        help="name of second camera"
    )

    args = parser.parse_args()

    return args



if __name__ == "__main__":

    args = parse_args()

    rgb_1, rgb_2 = create_train_dataset(pic_num=args.index, camera_name_1=args.camera_1, camera_name_2=args.camera_2)

    for model_name in args.models:
        
        matrix = get_matrix(rgb_1, rgb_2, model_name)

        for img_num in args.index: 
            number_str = '0' * (4 - len(str(img_num))) + str(img_num)

            sample = Image.open(f'dataset\{args.camera_1}_JPG\JPG\{args.camera_1}_{number_str}.jpg')

            lin_img = srgb_to_linear(np.array(sample))
            new_img = np.full(lin_img.shape, None)

            for i in range(len(lin_img)):
                new_img[i] = apply_matrix(lin_img[i], matrix)

            result_image = Image.fromarray(linear_to_srgb(new_img))

            result_image.save(f'{args.output_dir}/{img_num}_{model_name}_{args.camera_1[5:]}_to_{args.camera_2[5:]}.jpg')