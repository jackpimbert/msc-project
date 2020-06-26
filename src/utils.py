import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import argparse
import numpy as np
import os
import tensorflow as tf

plt.ioff()

parser = argparse.ArgumentParser(description='Run cGAN')
parser.add_argument('--name', default='cgan')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--loss-gan', type=float, default=0.01)
parser.add_argument('--loss-w2', type=float, default=1.0)
parser.add_argument('--loss-tv', type=float, default=1e-5)
parser.add_argument('--adam-lr', type=float, default=0.0002)
parser.add_argument('--adam-b1', type=float, default=0.5)
parser.add_argument('--batch-norm', default=False, action='store_true')
parser.add_argument('--skip-connections', default=False, action='store_true')
parser.add_argument('--dropout', default=False, action='store_true')
parser.add_argument('--do-flipping', default=False, action='store_true')
# Use upsampling instead of deconvolution/transposed convolution
parser.add_argument('--upsample', default=False, action='store_true')
parser.add_argument('--use-mask', default=False, action='store_true')
parser.add_argument('--target-mask', default=False, action='store_true')
# Training and testing arguments
parser.add_argument('--no-train', default=False, action='store_true')
parser.add_argument('--no-test', default=False, action='store_true')
# Generate a final dataset from the test data
parser.add_argument('--gen-dataset', default=False, action='store_true')

dataset_group = parser.add_mutually_exclusive_group(required=True)
dataset_group.add_argument('--mn40s', default=False, action='store_true')
# Target both mask and depth map, or depth map only, switch with use-mask.
dataset_group.add_argument('--mn40s-mask', default=False, action='store_true')
dataset_group.add_argument('--mn40s-bg', default=False, action='store_true')
dataset_group.add_argument('--mn40s-cropped', default=False, action='store_true')
dataset_group.add_argument('--mn40s-pivot', default=False, action='store_true')
dataset_group.add_argument('--net3d', default=False, action='store_true')

width_group = parser.add_mutually_exclusive_group(required=True)
width_group.add_argument('--w128', default=False, action='store_true')
width_group.add_argument('--w256', default=False, action='store_true')

loss_group = parser.add_mutually_exclusive_group(required=True)
loss_group.add_argument('--l1', default=False, action='store_true')
loss_group.add_argument('--l2', default=False, action='store_true')
loss_group.add_argument('--huber', default=False, action='store_true')
loss_group.add_argument('--berhu', default=False, action='store_true')


def parse():
    """ Generate the network name, given the program arguments."""
    args = parser.parse_args()

    if args.use_mask and args.target_mask:
        raise RuntimeError("Cannot target mask as output, and use mask!")

    bs = f"bs_{args.batch_size}"
    adam = "adam_{0}_{1}".format(str(args.adam_lr)[2:], str(args.adam_b1)[2:])
    bn = "_bn" if args.batch_norm else ""
    skip = "_skip" if args.skip_connections else ""
    dropout = "_dropout" if args.dropout else ""
    width = "_256" if args.w256 else "_128"
    mask = "_mask" if args.use_mask else ""
    target_mask = "_target_mask" if args.target_mask else ""
    upsample = "_upsample" if args.upsample else ""
    flip = "_flipping" if args.do_flipping else ""

    if args.l1:
        loss = "l1"
    elif args.l2:
        loss = "l2"
    elif args.huber:
        loss = "huber"
    elif args.berhu:
        loss = "berhu"

    lw1 = "lw_gan_{0}".format(int(args.loss_gan*100))
    lw2 = "lw_{1}_{0}".format(int(args.loss_w2*100), loss)
    lw3 = "lw_tv_{0}".format(int(args.loss_tv*100))

    if args.mn40s:
        data = "mn40s"
    elif args.mn40s_mask:
        data = "mn40s_mask"
    elif args.mn40s_bg:
        data = "mn40s_bg"
    elif args.mn40s_cropped:
        data = "mn40s_cropped"
    elif args.mn40s_pivot:
        data = "mn40s_pivot"
    elif args.net3d:
        data = "3dnet"
    else:
        data = "all"

    return f"{args.name}{width}_{data}{mask}{target_mask}_{bs}_{lw1}_{lw2}_{lw3}_{adam}{bn}{skip}{dropout}{upsample}{flip}", args


def _deprocess_image(image):
    """Deprocess an image to between 0/1 from -1/1."""
    return (image + 1)/2

def _process_image(image):
    return (image * 2) - 1

def _save_image(image, filepath):
    image_scaled = np.uint8(image*255)
    cv2.imwrite(filepath, image_scaled)

def setup_dirs(network_name):
    models_dir = './models'
    network_dir = os.path.join(models_dir, network_name)
    results_dir = os.path.join(network_dir, 'results')
    ckpt_dir = os.path.join(network_dir, 'checkpoint')
    graph_dir = os.path.join(network_dir, 'graph')
    dataset_dir = os.path.join(network_dir, 'dataset')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    return network_dir, ckpt_dir, results_dir, graph_dir, dataset_dir

def save_batch(images, output_dir, filenames):
    """Save the network output to disk, one batch at a time."""
    os.makedirs(output_dir, exist_ok=True)
    _, _, num_channels = images[0].shape
    for image in images:
        filename = os.path.basename(filenames.pop(0))
        filename = filename.replace("_noisy", "o_noisy")
        filename = filename.replace("_mask", "o_mask")
        for c in range(num_channels):
            img = _deprocess_image(image[:,:,c])
            img_name = filename if c == 0 else filename.replace("noisy", "mask")
            img_path = os.path.join(output_dir, img_name)
            _save_image(img, img_path)

def save_figure(images, batch_size, row_size, results_dir, filename, save_images=True):
    """Save the results of a batch as a figure, optionally save each image."""
    if save_images:
        images_dir = os.path.join(results_dir, 'images', filename, "depth")
        mask_dir = os.path.join(results_dir, 'images', filename, "mask")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

    _, _, num_channels = images[0].shape
    for c in range(num_channels):
        fig = plt.figure(figsize=(batch_size, row_size))
        gs = gridspec.GridSpec(batch_size, row_size)
        gs.update(wspace=0.05, hspace=0.05)
        row_count = 0
        for i, image in enumerate(images):
            if i % row_size == 0:
                row_count += 1

            image = _deprocess_image(image[:,:,c])

            # Plot in a grid
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')

            plt.imshow(image, cmap='gray')
            if save_images:
                target_dir = images_dir if c == 0 else mask_dir
                ii = i % 3
                filepath = os.path.join(target_dir, f"{row_count}_{ii}.png")
                _save_image(image, filepath)

        tag = "depth" if c == 0 else "mask"
        full_filename = f"{filename}-{tag}.png"
        plt.savefig(os.path.join(results_dir, full_filename))
        plt.close(fig)
        print(f"Figure '{full_filename}' saved to results.")

def load(session, saver, ckpt_dir):
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        global_step = tf.train.get_global_step()
        _global_step = session.run(global_step)
        print(f"Loading model from global step {_global_step}")
        return _global_step

    return 0

def save(session, saver, ckpt_dir, network_name, global_step):
    ckpt_name = os.path.join(ckpt_dir, network_name)
    saver.save(session, ckpt_name, global_step=global_step)
