import numpy as np
import os
import tensorflow as tf
import time
from hyperdash import Experiment

from src.datasets import (
    create_dataset,
    create_split_dataset,
)
from src.network import (
    log,
    generator,
    discriminator,
)
from src.utils import (
    load,
    parse,
    save,
    save_batch,
    save_figure,
    setup_dirs,
)
from src.losses import (
    berhu_loss,
)

network_name, args = parse()
print(f"Network: {network_name}")

network_dir, ckpt_dir, results_dir, graph_dir, dataset_dir = setup_dirs(network_name)
exp = Experiment(network_name)
log_file = os.path.join(network_dir, "log.txt")
with open(log_file, "a") as logging:
    logging.write(network_name + "\n")

# Network parameters
EPOCHS = exp.param("epochs", args.epochs)
BATCH_SIZE = exp.param("batch_size", args.batch_size)
DO_BN = exp.param("do_bn", args.batch_norm)
DO_SKIP = exp.param("do_skip", args.skip_connections)
DO_DROPOUT = exp.param("do_dropout", args.dropout)
# Adam optimiser parameters
ADAM_LR = exp.param("adam_lr", args.adam_lr)
ADAM_B1 = exp.param("adam_b1", args.adam_b1)
# Secondary loss function
LOSS_L1 = exp.param("loss_l1", args.l1)
LOSS_L2 = exp.param("loss_l2", args.l2)
LOSS_HUBER = exp.param("loss_huber", args.huber)
LOSS_BERHU = exp.param("loss_berhu", args.berhu)
# Weighting for generator loss
LOSS_GAN = exp.param("loss_gan", args.loss_gan) # GAN Loss
LOSS_W2 = exp.param("loss_w2", args.loss_w2) # Func loss
LOSS_TV = exp.param("loss_tv", args.loss_tv) # TV loss
# Input width
IN_WIDTH = exp.param("in_width", 256 if args.w256 else 128)
IN_HEIGHT = exp.param("in_height", IN_WIDTH)
IN_CHANNELS = exp.param("in_channels", 2 if args.use_mask else 1)

# Datasets
def get_data_iterator(data, prefetch=True):
    data_batched = data.batch(BATCH_SIZE)
    if prefetch:
        data_batched = data_batched.prefetch(buffer_size=PREFETCH_BUFFER_SIZE)
    iterator = data_batched.make_one_shot_iterator()
    return iterator.get_next()

# train_data, test_data, dataset_info = create_dataset(dataset_dir)
# train_data_count = len(dataset_info['input_train'])
# test_data_count = len(dataset_info['input_test'])

train_data, train_dataset_info = create_split_dataset(dataset_dir, train=True)
test_data, test_dataset_info = create_split_dataset(dataset_dir, train=False)
train_data_count = len(train_dataset_info['input_train'])
test_data_count = len(test_dataset_info['input_test'])

# Limit jobs to EPOCHS, even on several runs
STEPS_PER_EPOCH = train_data_count/BATCH_SIZE
PREFETCH_BUFFER_SIZE = BATCH_SIZE*50

inputs = tf.placeholder(tf.float32, [None, IN_HEIGHT, IN_WIDTH, IN_CHANNELS], "inputs")
targets = tf.placeholder(tf.float32, [None, IN_HEIGHT, IN_WIDTH, IN_CHANNELS], "targets")

outputs = generator(inputs, is_train=True, do_bn=DO_BN, do_skip=DO_SKIP, do_dropout=DO_DROPOUT)
outputs_test = generator(inputs, is_train=False, do_bn=DO_BN, do_skip=DO_SKIP, do_dropout=DO_DROPOUT)
predict_real = discriminator(inputs, targets, is_real=True, do_bn=DO_BN)
predict_fake = discriminator(inputs, outputs, is_real=False, do_bn=DO_BN)

# Discriminator loss
d_loss = -(tf.reduce_mean(log(predict_real) + log(1. - predict_fake)))
# Generator loss
g_loss_gan = -tf.reduce_mean(log(predict_fake))
# Total variational loss
g_loss_tv = tf.reduce_sum(tf.image.total_variation(outputs))

# Combined generator loss
if LOSS_L1:
    g_loss_func = tf.reduce_mean(tf.abs(outputs - targets))
    g_loss = LOSS_GAN*g_loss_gan + LOSS_W2*g_loss_func + LOSS_TV*g_loss_tv
if LOSS_L2:
    g_loss_func = tf.reduce_mean(tf.square(outputs - targets))
    g_loss = LOSS_GAN*g_loss_gan + LOSS_W2*g_loss_func + LOSS_TV*g_loss_tv
if LOSS_HUBER:
    g_loss_func = tf.losses.huber_loss(targets, outputs)
    g_loss = LOSS_GAN*g_loss_gan + LOSS_W2*g_loss_func + LOSS_TV*g_loss_tv
if LOSS_BERHU:
    g_loss_func = berhu_loss(targets, outputs)
    g_loss = LOSS_GAN*g_loss_gan + LOSS_W2*g_loss_func + LOSS_TV*g_loss_tv

mean_g_loss, mean_g_loss_op = tf.contrib.metrics.streaming_mean(g_loss)
mean_d_loss, mean_d_loss_op = tf.contrib.metrics.streaming_mean(d_loss)

# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if var.name.startswith("discriminator")]
g_vars = [var for var in t_vars if var.name.startswith("generator")]

# Optimise
global_step = tf.train.create_global_step()
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    d_train = tf.train.AdamOptimizer(ADAM_LR, ADAM_B1).minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_train = tf.train.AdamOptimizer(ADAM_LR, ADAM_B1).minimize(g_loss, var_list=g_vars)

#Create a saver object which will save all the variables
saver = tf.train.Saver(max_to_keep=1)

with tf.name_scope("summaries"):
    # Standard losses
    tf.summary.scalar("discrm. loss", d_loss)
    tf.summary.scalar("gen. loss GAN", g_loss_gan)
    tf.summary.scalar("gen. loss func", g_loss_func)
    tf.summary.scalar("gen. loss combined", g_loss)
    # Mean losses
    tf.summary.scalar("mean discrm. loss", mean_d_loss_op)
    tf.summary.scalar("mean gen. loss combined", mean_g_loss_op)
    # Histograms
    tf.summary.histogram("discrim. loss histogram", d_loss)
    tf.summary.histogram("gen.loss histogram", g_loss)
    summary_op = tf.summary.merge_all()

TRAIN = not args.no_train
TEST = not args.no_test
GENERATE_DATASET = args.gen_dataset
with tf.Session() as sess:
    _global_step = load(sess, saver, ckpt_dir)

    if TRAIN:
        print("Training...")
        # Write a graph event so we can use tensorboard to see the structure of the network
        writer = tf.summary.FileWriter(graph_dir, sess.graph)

        for epoch in range(EPOCHS):
            global_epoch = np.ceil(_global_step/STEPS_PER_EPOCH)
            if global_epoch >= EPOCHS:
                print(f"Max number of global epochs reached ({global_epoch}/{EPOCHS}).")
                break

            print(f"epoch: {epoch}, global epoch: {global_epoch}, global step: {_global_step}")
            next_training_pair = get_data_iterator(train_data, prefetch=True)

            dl = 0.
            gl = 0.
            itr = 0
            last_itr_count = 0
            start = time.time()
            while True:
                try:
                    input_batch, target_batch = sess.run(next_training_pair)
                except tf.errors.OutOfRangeError:
                    break

                _, dl = sess.run(
                    [d_train, d_loss],
                    feed_dict={inputs: input_batch, targets: target_batch})

                _, gl = sess.run(
                    [g_train, g_loss],
                    feed_dict={inputs: input_batch, targets: target_batch})

                _global_step, summary, mean_dl, mean_gl = sess.run(
                    [global_step, summary_op, mean_d_loss, mean_g_loss],
                    feed_dict={inputs: input_batch, targets: target_batch})

                itr += 1
                print(f"batch: {itr}, global epoch: {global_epoch}, global step: {_global_step}")
                exp.metric("dl", dl)
                exp.metric("gl", gl)
                exp.metric("mean dl", mean_dl)
                exp.metric("mean gl", mean_gl)
                writer.add_summary(summary, global_step=_global_step)

                if _global_step % 500 == 0:
                    generated_batch = sess.run(outputs, feed_dict={inputs: input_batch, targets: target_batch})
                    images = list(sum(zip(input_batch, target_batch, generated_batch), ()))
                    filename = f"training_step-{_global_step}"
                    save_figure(images, BATCH_SIZE, 3, results_dir, filename)

            # Collect time stats
            end = time.time()
            total = end - start
            itr_sec = (itr - last_itr_count)/total
            last_itr_count = itr
            start = time.time()

            epoch_summary = "epoch {0} (global epoch {1}/{2} [{3}]) complete. {4:.2f} itr/sec".format(epoch, global_epoch,  EPOCHS, _global_step, itr_sec)
            print(epoch_summary)

            # log to file
            with open(log_file, "a") as logging:
                logging.write(epoch_summary + "\n")

            # save model
            save(sess, saver, ckpt_dir, network_name, _global_step)

    if TEST:
        print("Testing...")
        _global_step = load(sess, saver, ckpt_dir)

        next_element_test = get_data_iterator(test_data, prefetch=False)
        for batch_idx in range(20):
            input_batch, target_batch = sess.run(next_element_test)
            generated_batch = sess.run(outputs_test, feed_dict={inputs: input_batch, targets: target_batch})
            # Zip images: (a1,a2) (b1,b2) (c1,c2) -> (a1,b1,c1,a2,b2,c2)
            images = list(sum(zip(input_batch, target_batch, generated_batch), ()))

            filename = f"test_b{batch_idx}_step-{_global_step}"
            save_figure(images, BATCH_SIZE, 3, results_dir, filename)

    if GENERATE_DATASET:
        print("Generating train dataset...")
        _global_step = load(sess, saver, ckpt_dir)

        filenames = train_dataset_info['input_train']
        next_element_train = get_data_iterator(train_data, prefetch=False)
        output_dir = os.path.join(dataset_dir, "train")
        while True:
            try:
                input_batch, target_batch = sess.run(next_element_train)
            except tf.errors.OutOfRangeError:
                break

            generated_batch = sess.run(outputs_test, feed_dict={inputs: input_batch, targets: target_batch})
            save_batch(generated_batch, output_dir, filenames)
            print(f"{len(filenames)}/{train_data_count}")

        filenames = test_dataset_info['input_test']
        next_element_test = get_data_iterator(test_data, prefetch=False)
        output_dir = os.path.join(dataset_dir, "test")
        while True:
            try:
                input_batch, target_batch = sess.run(next_element_test)
            except tf.errors.OutOfRangeError:
                break

            generated_batch = sess.run(outputs_test, feed_dict={inputs: input_batch, targets: target_batch})
            save_batch(generated_batch, output_dir, filenames)
            print(f"{len(filenames)}/{test_data_count}")

exp.end()
