import tensorflow as tf
import glob
import socket
import numpy as np
import os
import random
from proj.utils import parse
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

RANDOM_STATE = 444
TEST_SPLIT = 0.33

_, args = parse()

IN_WIDTH = 256 if args.w256 else 128
IN_HEIGHT = IN_WIDTH
IM_CHANNELS = 1
IM_MAX_VALUE = 255
USE_MASK = args.use_mask
TARGET_MASK = args.target_mask
DO_FLIPPING = args.do_flipping

# Synthetic ModelNet40 Dataset
MN40S = 'Synthetic/ModelNet40'
MN40S_MASK = 'Synthetic/ModelNet40_MASK'
MN40S_BG = 'Synthetic/ModelNet40_MASK_BG'
MN40S_CROP = 'Synthetic/ModelNet40_MASK_BG_CROP'
MN40S_PIVOT = 'Synthetic/ModelNet40_MASK_BG_PIVOT_CROP'
NET3D = 'Synthetic/3DNet'

hostname = socket.gethostname().lower()
if hostname.startswith('jack') or hostname.startswith('anon'):
    ROOT_FOLDER = '/Volumes/Seagate macOS/Project/DATA'
else:
    ROOT_FOLDER = '/home/s1027418/datasets'

DATA_FOLDERS = [MN40S]


def create_split_dataset(dataset_dir, train=True):
    data_folders = _get_data_folders()
    split_folder = "train" if train else "test"

    combined_input_files = []
    combined_target_files = []
    for data_folder in data_folders:
        input_files, target_files = _create_dataset(data_folder, split_folder)
        combined_input_files += input_files
        combined_target_files += target_files

    input_split, target_split = shuffle(
            combined_input_files,
            combined_target_files,
            random_state=RANDOM_STATE)

    tag = 'train' if train else 'test'
    split_dataset_info = {
        f'input_{tag}': input_split,
        f'target_{tag}': target_split,
    }
    _write_dataset_info(split_dataset_info, dataset_dir)

    # Prepare for tensorflow dataset
    input_split = tf.constant(input_split)
    target_split = tf.constant(target_split)

    # Create tensors
    split_tensor = tf.data.Dataset.from_tensor_slices((input_split, target_split))

    # Parse images
    split_data = split_tensor.map(_parse_function)

    return split_data, split_dataset_info

def create_dataset(dataset_dir):
    data_folders = _get_data_folders()

    combined_input_files = []
    combined_target_files = []
    for data_folder in data_folders:
        input_files, target_files = _create_dataset(data_folder)
        combined_input_files += input_files
        combined_target_files += target_files

    # Split into training and testing
    input_train, input_test, target_train, target_test = train_test_split(
            combined_input_files,
            combined_target_files,
            test_size=TEST_SPLIT,
            random_state=RANDOM_STATE,
            shuffle=True)

    dataset_info = {
        'input_train': input_train,
        'input_test': input_test,
        'target_train': target_train,
        'target_test': target_test,
    }
    _write_dataset_info(dataset_info, dataset_dir)

    # Prepare for tensorflow dataset
    input_train = tf.constant(input_train)
    input_test = tf.constant(input_test)
    target_train = tf.constant(target_train)
    target_test = tf.constant(target_test)

    # Create tensors
    train_tensor = tf.data.Dataset.from_tensor_slices((input_train, target_train))
    test_tensor = tf.data.Dataset.from_tensor_slices((input_test, target_test))

    # Parse images
    train_data = train_tensor.map(_parse_function)
    test_data = test_tensor.map(_parse_function)

    return train_data, test_data, dataset_info

def _get_data_folders():
    if args.mn40s:
        return [MN40S]
    elif args.mn40s_mask:
        return [MN40S_MASK]
    elif args.mn40s_bg:
        return [MN40S_BG]
    elif args.mn40s_cropped:
        return [MN40S_CROP]
    elif args.mn40s_pivot:
        return [MN40S_PIVOT]
    elif args.net3d:
        return [NET3D]

    return DATA_FOLDERS

def _create_dataset(data_folder, split_folder="scene_*"):
    input_path = os.path.join(ROOT_FOLDER, data_folder, "original", split_folder, "**/depth/*_noisy.png")
    if TARGET_MASK:
        target_path = os.path.join(ROOT_FOLDER, data_folder, "original", split_folder, "**/kinect/*_mask.png")
    else:
        target_path = os.path.join(ROOT_FOLDER, data_folder, "original", split_folder, "**/kinect/*_noisy.png")

    # List of filenames
    input_filenames = sorted(glob.glob(input_path, recursive=True))
    target_filenames = sorted(glob.glob(target_path, recursive=True))

    num_input = len(input_filenames)
    num_target = len(target_filenames)

    print(f"Processing dataset: {data_folder} [{num_input}/{num_target}]")

    return input_filenames, target_filenames

def _parse_function(input, target):
    seed = random.randint(0, 2**31 - 1)
    input_centered = _preprocess_image(input, IM_CHANNELS, seed)
    target_centered = _preprocess_image(target, IM_CHANNELS, seed)

    return input_centered, target_centered

def _preprocess_image(image, channels, seed):
    """ Preprocess images from dataset.

    - Resize to global IN_HEIGHT, IN_WIDTH
    - Scale and center as preprocessing step to between -1/1.
    """
    image_decoded = _load_image(image, channels)
    image_centered = _scale_and_center_image(image_decoded)

    if USE_MASK:
        # Get the length of the image path string
        image_len = tf.size(tf.string_split([image],""))
        # Get the path up to "noisy.png"
        image_base = tf.substr(image, 0, image_len-9)
        # Replace "noisy.png" with "mask.png"
        mask = tf.add(image_base, tf.constant('mask.png'))
        # Load the valid/invalid pixel mask
        # 1: valid (after preprocessing)
        # -1: invalid (after preprocessing)
        mask_decoded = _load_image(mask, 1)
        mask_centered = _scale_and_center_image(mask_decoded)

        # Now append mask to image_centered as another channel
        image_centered = tf.concat([image_centered, mask_centered], axis=2)

    if DO_FLIPPING:
        image_centered = tf.image.random_flip_left_right(image_centered, seed=seed)

    return image_centered

def _load_image(image, channels):
    image_string = tf.read_file(image)
    image_decoded = tf.image.decode_png(image_string, channels)

    return image_decoded

def _scale_and_center_image(image):
    # Crop and resize, then scale from 0-65535 to 0-1, finally center to -1,1
    if args.mn40s_cropped or args.mn40s_pivot or args.net3d:
        # We're already square, at 256x256 pixels.
        image_resized = image
    else:
        # Square the input image via cropping
        image_resized = tf.image.resize_image_with_crop_or_pad(image, 480, 480)

    image_resized = tf.image.resize_images(image_resized, [IN_HEIGHT, IN_WIDTH])
    image_scaled = tf.divide(image_resized, IM_MAX_VALUE)
    image_mul = tf.multiply(image_scaled, 2)
    image_centered = tf.subtract(image_mul, 1)

    return image_centered

def _write_dataset_info(dataset_info, dataset_dir):
    for name, data in dataset_info.items():
        filename = os.path.join(dataset_dir, f"{name}.txt")
        with open(filename, 'w') as f:
            f.write("\n".join(data))
