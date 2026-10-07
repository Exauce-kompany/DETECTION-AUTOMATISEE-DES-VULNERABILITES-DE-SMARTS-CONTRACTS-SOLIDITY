"""TensorFlow input batches and ordered prediction for the production model."""

import numpy as np

from src.preprocessing import as_windows


def make_dataset(arrays, config, weights=None, training=False, seed=42):
    import tensorflow as tf

    window_size = config["window_size"]

    def generate():
        for i, label in enumerate(arrays["labels"]):
            start, end = arrays["offsets"][i : i + 2]
            inputs = as_windows(arrays["tokens"][start:end], window_size)
            item = (inputs, np.asarray([label], dtype=np.float32))
            yield (*item, np.float32(weights[i])) if weights is not None else item

    signature = (tf.TensorSpec((None, window_size), tf.int32), tf.TensorSpec((1,), tf.float32))
    shapes = ([None, window_size], [1])
    if weights is not None:
        signature += (tf.TensorSpec((), tf.float32),)
        shapes += ([],)
    dataset = tf.data.Dataset.from_generator(generate, output_signature=signature)
    if training:
        dataset = dataset.shuffle(
            min(len(arrays["labels"]), 4096), seed=seed, reshuffle_each_iteration=True
        )
        batch = config["batch_size"]
        dataset = dataset.bucket_by_sequence_length(
            lambda *items: tf.shape(items[0])[0],
            [4, 8, 16, 32, 64],
            [batch, batch, max(8, batch // 2), max(4, batch // 4), 4, 2],
            padded_shapes=shapes,
        )
    else:
        dataset = dataset.padded_batch(config["batch_size"], padded_shapes=shapes)
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 2
    return dataset.with_options(options).prefetch(1)


def predict_logits(model, arrays, config):
    # Preserve original record ordering; prediction datasets are never bucketed/shuffled.
    dataset = make_dataset(arrays, config).map(lambda x, y: x)
    return model.predict(dataset, verbose=0).reshape(-1)
