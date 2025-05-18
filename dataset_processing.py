from PIL import Image
import numpy as np
import numpy.typing as npt
from typing import Literal, List
import matplotlib.pyplot as plt


def create_train_dataset(
    pic_num : List[int],
    camera_name_1 : str = 'Canon1DsMkIII',
    camera_name_2 : str = 'Canon600D'
        
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """
    This extracts all colors from colorcheckers from 'dataset/Canon1DsMkIII_JPG' and 'dataset/Canon600D_JPG'.

    Parameters
    ----------
    pic_num : int
        Picture index in dataset (from 1 to 197)

    camera_name_1 : str
        First camera name. Optional. Is used if you want to swap cameras or for debug.

    camera_name_2 : str
        Second camera name. Optional. Is used if you want to swap cameras or for debug.

    Returns
    -------
    (rgb_1, rgb_2) : tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]
        Dataset of colors for first camera and corresponding them colors for second camera.
    """
    for i in range(len(pic_num)):
        if i == 0:
            sRGB_patches_1 = crop_color_patches(camera_name_1, pic_num[i])
            sRGB_patches_2 = crop_color_patches(camera_name_2, pic_num[i])

        else:
            sRGB_patches_1.extend(crop_color_patches(camera_name_1, pic_num[i]))
            sRGB_patches_2.extend(crop_color_patches(camera_name_2, pic_num[i]))
            
        rgb_1 = srgb_to_linear(extract_avg_colors(sRGB_patches_1))
        rgb_2 = srgb_to_linear(extract_avg_colors(sRGB_patches_2))
    
    return rgb_1, rgb_2


def extract_avg_colors(
    patch_list : npt.NDArray[np.int_]
) -> npt.NDArray[np.int_]:
    """
    This function extracts average color from color patch.

    Parameters
    ----------
    patch_list : npt.NDArray[np.int_]
        Array of colors of patch.

    Returns
    -------
    color : npt.NDArray[np.int_]
        Average color of patch.
    """

    for i, c in enumerate(patch_list):

        color = np.round(np.array(c).sum(axis=0).sum(axis=0, keepdims=True) / (np.array(c).size // 3)).astype(int)

        if i == 0:
            colors = np.array(color)
        else:
            colors = np.append(colors, color, axis = 0)

    return colors


def crop_color_patches(
    camera_name : Literal['Canon1DsMkIII', 'Canon600D'],
    pic_number : int,
    show_ROI : bool = False,
    show_patches : bool = False,
    image_path : str = None

) -> List[npt.NDArray[np.float_]]:
    """
    This function gets cropped color patches from image in dataset.

    Parameters
    ----------
    camera_name : Literal['Canon1DsMkIII', 'Canon600D']
        Camera name, available options are ['Canon1DsMkIII', 'Canon600D'].

    pic_number : int
        Picture index in dataset.

    picture_path : str
        Path to picture in dataset (optional argument). 
        If it is None camera_name and pic_number are used. 
        If not None it is uset to get picture from dataset and mask for it.

    show_ROI : bool
        If True ROI (colorchecker) is showed. Default is False.

    show_patches : bool
        If True all color patches from colorchecker are showed. Default is False.

    image_path : str
        Path to image. If not None is used to read image.

    Returns
    -------
    patches : list
        List of all color patches from colorchecker.

    """

    assert camera_name in ['Canon1DsMkIII', 'Canon600D']

    # number in same format as in dataset
    number_str = '0' * (4 - len(str(pic_number))) + str(pic_number)

    # Reading mask for all patches
    if image_path is None:
        sample = Image.open(f'dataset\{camera_name}_JPG\JPG\{camera_name}_{number_str}.jpg')
    else:
        sample = Image.open(image_path)

    with open(f'dataset\{camera_name}_CHECKER\{camera_name}_{number_str}_mask.txt', 'r') as f:
        lines = f.readlines()

    # ROI - Region Of Interest, colorchecker in our images. Multiplicated on 2 to account for differences in JPEG and PNG
    roi = np.array([float(i) for i in lines[0][:-1].split(',')]) * 2

    # Indices of rotated images in dataset. Selected manually from dataset
    ind_rotated = [3, 7, 10, 27, 30, 31, 34, 37, 39, 41, \
               46, 48, 52, 53, 59, 63, 64, 66, 67, 68, \
                70, 75, 76, 78, 83, 84, 87, 96, 101, 103, \
                104, 105, 110, 114, 118, 121, 122, 129, \
                130, 131, 132, 134, 137, 138, 141, 142, 143, \
                147, 149, 150, 151, 156, 164, 172, 175, 193]

    if pic_number in ind_rotated:
        rotated = True
    else:
        rotated = False

    if not rotated:
        sample_ROI = sample.crop([roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]])
    else:
        sample_ROI = sample.crop([roi[1], sample.size[1] - roi[0] - roi[2], roi[1] + roi[3], sample.size[1] - roi[0]])
    sample_ROI_size = sample_ROI.size

    if show_ROI:
        plt.imshow(sample_ROI)
        plt.axis('off')
        plt.show()

    patches = []
    for i in range(1, 49, 2):
        # Coordinates of color patches. Multiplicated on 2 to account for differences in JPEG and PNG
        x_coords = np.array([float(i) for i in lines[i][:-1].split(',')]) * 2 
        y_coords = np.array([float(i) for i in lines[i+1][:-1].split(',')]) * 2 
        if not rotated:
            coords = [x_coords[0], y_coords[0], x_coords[2], y_coords[2]]
        else:
            coords = [y_coords[0], sample_ROI_size[1] - x_coords[3], y_coords[2], sample_ROI_size[1] - x_coords[1]]

        if show_patches:
            plt.imshow(sample_ROI.crop(coords))
            plt.axis('off')
            plt.show()

        patches.append(np.array(sample_ROI.crop(coords)))

    return patches


def srgb_to_linear(
    image_srgb_array : npt.NDArray[np.int_]

) -> npt.NDArray[np.float_]:
    """
    This function converts image sRGB coordinates from srgb to linear coordinates.

    Parameters
    ----------
    image_srgb_array : npt.NDArray[np.float_]
        Image array in sRGB.

    Returns
    -------
    image_linear_array : npt.NDArray[np.float_]
        Image array in linear coordinates.

    """
    srgb_normalized = image_srgb_array / 255.0

    threshold = 0.04045
    a = 0.055
    gamma = 2.4
    
    image_linear_array = np.where(
        srgb_normalized > threshold,
        ((srgb_normalized + a) / (1 + a)) ** gamma,
        srgb_normalized / 12.92
    )
    return image_linear_array


def linear_to_srgb(
    image_linear_array : npt.NDArray[np.float_]

) -> npt.NDArray[np.int_]:
    """
    This function converts image linear coordinates to sRGB coordinates.

    Parameters
    ----------
    image_linear_array : npt.NDArray[np.float_]
        Image array in linear coordinates.
    
    Returns
    -------
    image_srgb_array : npt.NDArray[np.float_]
        Image array in sRGB.
    """

    srgb = np.clip(image_linear_array.astype(np.float64), 0.0, 1.0).copy()
    
    mask = srgb > 0.0031308
    srgb[mask] = 1.055 * (srgb[mask] ** (1/2.4)) - 0.055
    srgb[~mask] *= 12.92
    
    return np.around(srgb * 255.0).astype(np.uint8)

