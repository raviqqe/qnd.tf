import qnd
import tensorflow as tf


def _preprocess_image(image):
    if(int(tf.__version__.split(".")[1])<13 and int(tf.__version__.split(".")[0])<2): ### For tf version < 1.13.0
        return tf.to_float(image) / 255 - 0.5
    else: ### For tf version >= 1.13.0
        return tf.cast(image,tf.float32) / 255 - 0.5
    

def read_file(filename_queue):
    _, serialized = tf.TFRecordReader().read(filename_queue)

    def scalar_feature(dtype): return tf.FixedLenFeature([], dtype)

    features = tf.parse_single_example(serialized, {
        "image_raw": scalar_feature(tf.string),
        "label": scalar_feature(tf.int64),
    })

    image = tf.decode_raw(features["image_raw"], tf.uint8)
    image.set_shape([28**2])

    return _preprocess_image(image), features["label"]


def serving_input_fn():
    features = {
        'image': _preprocess_image(tf.placeholder(tf.uint8, [None, 28**2])),
    }

    return tf.contrib.learn.InputFnOps(features, None, features)


def minimize(loss):
    if(int(tf.__version__.split(".")[1])<4 and int(tf.__version__.split(".")[0])==1) or int(tf.__version__.split(".")[0]) == 0: ### for tf version <1.4.0
        return tf.train.AdamOptimizer().minimize(
            loss,
            tf.contrib.framework.get_global_step())
    else: ###for version >= 1.4.0
        return tf.train.AdamOptimizer().minimize(
            loss,
            tf.train.get_global_step())


def def_model():
    qnd.add_flag("hidden_layer_size", type=int, default=64,
                 help="Hidden layer size")

    def model(image, number=None, mode=None):
        h = tf.contrib.layers.fully_connected(image,
                                              qnd.FLAGS.hidden_layer_size)
        h = tf.contrib.layers.fully_connected(h, 10, activation_fn=None)

        predictions = tf.argmax(h, axis=1)

        if mode == tf.contrib.learn.ModeKeys.INFER:
            return predictions

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=number,
                                                           logits=h))
        if(int(tf.__version__.split(".")[1])<5 and int(tf.__version__.split(".")[0])==1) or int(tf.__version__.split(".")[0]) == 0: ### for tf version <1.5.0
            return predictions, loss, minimize(loss), {
                "accuracy": tf.contrib.metrics.streaming_accuracy(predictions,
                                                                number)[1],
            }
        else: ###for version >= 1.5.0 the new api changed signature sequence
            return predictions, loss, minimize(loss), {
                "accuracy": tf.metrics.accuracy(number,
                                                predictions)[1],
            } 
    return model
