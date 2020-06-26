import tensorflow as tf

def berhu_loss(labels, predictions, scope=None):
    """Adds a (Reverse) Huber Loss term to the training procedure."""
    if labels is None:
        raise ValueError("labels must not be None.")
    if predictions is None:
        raise ValueError("predictions must not be None.")

    with tf.name_scope(scope, "berhu_loss",
                       (predictions, labels)) as scope:
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        error = tf.subtract(predictions, labels)
        abs_error = tf.abs(error)

        c = 0.2 * tf.reduce_max(abs_error)

        berhu_loss = tf.where(abs_error <= c,
                             abs_error,
                             (tf.square(error) + tf.square(c))/(2*c))
        loss = tf.reduce_mean(berhu_loss)

    return loss
