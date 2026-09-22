"""Hierarchical classifier: local token convolutions, then a BiLSTM over windows."""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="SmartBug")
class WindowEncoder(tf.keras.layers.Layer):
    def __init__(self, vocabulary_size, embedding_dim, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_dim)
        self.convolution = tf.keras.layers.Conv1D(filters, kernel_size, padding="same", activation="relu")

    def build(self, input_shape):
        self.embedding.build((None, input_shape[-1]))
        self.convolution.build((None, input_shape[-1], self.embedding_dim))
        super().build(input_shape)

    def call(self, inputs):
        shape = tf.shape(inputs)
        flat = tf.reshape(inputs, (-1, shape[-1]))
        mask = tf.not_equal(flat, 0)[..., None]
        embedded = self.embedding(flat) * tf.cast(mask, self.compute_dtype)
        features = self.convolution(embedded)
        count = tf.reduce_sum(tf.cast(mask, features.dtype), axis=1)
        mean = tf.reduce_sum(features * tf.cast(mask, features.dtype), axis=1) / tf.maximum(count, 1.0)
        maximum = tf.reduce_max(tf.where(mask, features, tf.cast(-1e9, features.dtype)), axis=1)
        maximum = tf.where(count > 0, maximum, tf.zeros_like(maximum))
        combined = tf.concat([mean, maximum], axis=-1)
        return tf.reshape(combined, (shape[0], shape[1], self.filters * 2))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.filters * 2)

    def get_config(self):
        return {**super().get_config(), "vocabulary_size": self.vocabulary_size, "embedding_dim": self.embedding_dim, "filters": self.filters, "kernel_size": self.kernel_size}


@tf.keras.utils.register_keras_serializable(package="SmartBug")
class WindowMask(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_any(tf.not_equal(inputs, 0), axis=-1)


@tf.keras.utils.register_keras_serializable(package="SmartBug")
class MaskedPool(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        values, mask = inputs
        mask = mask[..., None]
        count = tf.reduce_sum(tf.cast(mask, values.dtype), axis=1)
        mean = tf.reduce_sum(values * tf.cast(mask, values.dtype), axis=1) / tf.maximum(count, 1.0)
        maximum = tf.reduce_max(tf.where(mask, values, tf.cast(-1e9, values.dtype)), axis=1)
        maximum = tf.where(count > 0, maximum, tf.zeros_like(maximum))
        return tf.concat([mean, maximum], axis=-1)


def build_model(config, vocabulary_size):
    inputs = tf.keras.Input(shape=(None, config["window_size"]), dtype="int32", name="contract_windows")
    mask = WindowMask(name="window_mask")(inputs)
    x = WindowEncoder(vocabulary_size, config["embedding_dim"], config["convolution_filters"], config["convolution_kernel"], name="window_encoder")(inputs)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config["lstm_units"], return_sequences=True), name="contract_bilstm")(x, mask=mask)
    x = MaskedPool(name="contract_pool")([x, mask])
    x = tf.keras.layers.Dropout(config["dropout"])(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    output = tf.keras.layers.Dense(1, name="vulnerability_logit")(x)
    model = tf.keras.Model(inputs, output, name="smart_bug_hierarchical_v3")
    model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0)])
    return model
