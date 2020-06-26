import tensorflow as tf
from proj.utils import parse

_, args = parse()

IN_WIDTH = 256 if args.w256 else 128
IN_HEIGHT = IN_WIDTH
IN_CHANNELS = 2 if args.use_mask else 1

EPS = 1e-12

def log(x):
    """Prevent numeric errors due to rounding."""
    return tf.log(tf.maximum(x, EPS))

def bnorm(x, training=True):
    return tf.layers.batch_normalization(x, momentum=0.9, training=training)

def dis_conv2d(x, filters, kernal_size, strides, activation, name, bn=True):
    """Conv2D for the discriminator network."""
    with tf.variable_scope(f"dis_{name}") as scope:
        # Pad the x,y dimensions of the [batch, x, y, d] input
        x_padded = tf.pad(x, [[0,0], [1,1], [1,1,], [0,0]], mode="CONSTANT")
        conv = tf.layers.conv2d(
            inputs=x_padded,
            filters=filters,
            kernel_size=kernal_size,
            padding="valid",
            strides=strides,
            activation=activation)
        if bn:
            return bnorm(conv)
        else:
            return conv

def conv2d(x, filters,  activation, is_train, name, bn=True):
    with tf.variable_scope(name) as scope:
        conv = tf.layers.conv2d(
            inputs=x,
            filters=filters,
            kernel_size=4,
            padding="same",
            strides=2,
            activation=activation)
        if bn:
            return bnorm(conv, training=is_train)
        else:
            return conv

def upsample2d(x, filters, activation, is_train, name, bn=True, dropout=0.0):
    with tf.variable_scope(name) as scope:
        _, h, w, _ = x.shape
        size = [h*2, w*2]
        resized = tf.image.resize_nearest_neighbor(x, size)
        conv = tf.layers.conv2d(
            inputs=resized,
            filters=filters,
            kernel_size=4,
            padding="same",
            activation=activation)

        if bn:
            conv = bnorm(conv, training=is_train)

        if dropout > 0.0:
            conv = tf.nn.dropout(conv, keep_prob=1-dropout)

        return conv

def deconv2d(x, filters, activation, is_train, name, bn=True, dropout=0.0):
    with tf.variable_scope(name) as scope:
        deconv = tf.layers.conv2d_transpose(
            inputs=x,
            filters=filters,
            kernel_size=4,
            padding="same",
            strides=2,
            activation=activation)

        if bn:
            deconv = bnorm(deconv, training=is_train)

        if dropout > 0.0:
            deconv = tf.nn.dropout(deconv, keep_prob=1-dropout)

        return deconv

def get_network_specs():
    if IN_WIDTH == 256:
        # Encoder stage, with 256x256xIN_CHANNELS input.
        init_channels = 32
        encoder_specs = [
            init_channels * 1, # 128x128x32
            init_channels * 2, # 64x64x64
            init_channels * 4, # 32x32x128
            init_channels * 8, # 16x16x256
            init_channels * 16, # 8x8x512
            init_channels * 16, # 4x4x512
            init_channels * 16, # 2x2x512
            init_channels * 16, # 1x1x512
        ]

        # Decoder stage
        decoder_specs = [
            init_channels * 16, # 2x2x512
            init_channels * 16, # 4x4x512
            init_channels * 16, # 8x8x512
            init_channels * 8, # 16x16x256
            init_channels * 4, # 32x32x128
            init_channels * 2, # 64x64x64
            init_channels * 1, # 128x128x32
            IN_CHANNELS, # 256x256xIN_CHANNELS
        ]
    else:
        # Encoder stage, with 128x128xIN_CHANNELS input.
        init_channels = 64
        encoder_specs = [
            init_channels * 1, # 64x64x64
            init_channels * 2, # 32x32x128
            init_channels * 4, # 16x16x256
            init_channels * 8, # 8x8x512
            init_channels * 8, # 4x4x512
            init_channels * 8, # 2x2x512
            init_channels * 8, # 1x1x512
        ]

        # Decoder stage
        decoder_specs = [
            init_channels * 8, # 2x2x512
            init_channels * 8, # 4x4x512
            init_channels * 8, # 8x8x512
            init_channels * 4, # 16x16x256
            init_channels * 2, # 32x32x128
            init_channels * 1, # 64x64x64
            IN_CHANNELS, # 128x128xIN_CHANNELS
        ]

    return encoder_specs, decoder_specs

def generator(inputs, is_train=True, do_bn=True, do_skip=True, do_dropout=True):
    """Generator.

    - Encoder/Decoder architecture, without skip connections.
    """
    dropout = 0.5 if do_dropout else 0.0
    encoder_specs, decoder_specs = get_network_specs()

    reuse = False if is_train else True
    with tf.variable_scope("generator", reuse=reuse):
        # Build encoder layers
        layers = []
        for idx, channels in enumerate(encoder_specs, start=1):
            tag = f"conv{idx}"
            prev = layers[-1] if idx > 1 else inputs

            if idx == len(decoder_specs):
                # Final layer
                conv = conv2d(prev, channels, tf.nn.leaky_relu, is_train, tag, bn=False)
            else:
                conv = conv2d(prev, channels, tf.nn.leaky_relu, is_train, tag, bn=do_bn)

            layers.append(conv)

        # Build decoder layers
        num_encoder_layers = len(layers)
        for idx, channels in enumerate(decoder_specs, start=1):
            tag = f"deconv{idx}"
            prev = layers[-1]
            if do_skip and idx > 1:
                # Skip connections start from the second decoder layer
                skip = layers[num_encoder_layers - idx]
                prev = tf.concat([prev, skip], axis=3)

            if idx == len(decoder_specs):
                # Final layer
                if args.upsample:
                    deconv = upsample2d(prev, channels, tf.nn.tanh, is_train, tag, bn=False)
                else:
                    deconv = deconv2d(prev, channels, tf.nn.tanh, is_train, tag, bn=False)
            elif idx <= 3 and dropout > 0.0:
                # First three layers have dropout
                deconv = deconv2d(prev, channels, tf.nn.relu, is_train, tag, bn=do_bn, dropout=dropout)
            else:
                # No dropout
                deconv = deconv2d(prev, channels, tf.nn.relu, is_train, tag, bn=do_bn)

            layers.append(deconv)

    return layers[-1]

def discriminator(inputs, outputs, is_real, do_bn=True):
    """Discriminator.

    - PatchGAN architecture, where each cell in the 30x30 output
      represents a patch of the inputs.
    - Input depth is 2, as we concat the generated map and the target map.
    """
    reuse = False if is_real else True
    with tf.variable_scope("discriminator", reuse=reuse):
        # Concat along 3rd axis to make 128x128x(2*IN_CHANNELS) input.
        joint_inputs = tf.concat([inputs, outputs], axis=3)

        if IN_WIDTH == 256:
	    # In 256x256x(2*IN_CHANNELS)
            conv1 = dis_conv2d(joint_inputs, 64, 4, 2, tf.nn.leaky_relu, f"conv1", bn=do_bn)
            # 128x128x64
            conv2 = dis_conv2d(conv1, 128, 4, 2, tf.nn.leaky_relu, f"conv2", bn=do_bn)
            # 64x64x128
            conv2 = dis_conv2d(conv1, 256, 4, 2, tf.nn.leaky_relu, f"conv3", bn=do_bn)
            # 32x32x256
            conv3 = dis_conv2d(conv2, 512, 4, 1, tf.nn.leaky_relu, f"conv4", bn=do_bn)
            # 31x31x512
            output = dis_conv2d(conv3, IN_CHANNELS, 4, 1, tf.nn.sigmoid, f"conv5", bn=False)
            # Out: 30x30xIN_CHANNELS
        else:
            # In 128x128x(2*IN_CHANNELS)
            conv1 = dis_conv2d(joint_inputs, 128, 4, 2, tf.nn.leaky_relu, f"conv1", bn=do_bn)
            # 64x64x128
            conv2 = dis_conv2d(conv1, 256, 4, 2, tf.nn.leaky_relu, f"conv2", bn=do_bn)
            # 32x32x256
            conv3 = dis_conv2d(conv2, 512, 4, 1, tf.nn.leaky_relu, f"conv3", bn=do_bn)
            # 31x31x512
            output = dis_conv2d(conv3, IN_CHANNELS, 4, 1, tf.nn.sigmoid, f"conv4", bn=False)
            # Out: 30x30xIN_CHANNELS

        return output
