import tensorflow as tf
from kshape.core import _sbd


class Encoder(tf.keras.Model):

    def __init__(self, input_shape, code_size, filters, kernel_sizes):
        super(Encoder, self).__init__()
        assert len(filters) == len(kernel_sizes)
        assert len(input_shape) == 2  # (x, y), x = # of samples, y = # of vars
        # self.input_shape = input_shape
        self.code_size = code_size

        self.convs = []
        self.norms = []
        output_len = input_shape[0]
        output_channels = input_shape[1]

        for f, k in zip(filters, kernel_sizes):
            l = tf.keras.layers.Conv1D(f, k, activation="tanh")
            b = tf.keras.layers.BatchNormalization()
            self.convs.append(l)
            self.norms.append(b)
            output_len = output_len - (k-1)
            output_channels = f

        self.last_kernel_shape = (output_len, output_channels)
        self.flatten = tf.keras.layers.Flatten()
        self.out = tf.keras.layers.Dense(code_size)

    def call(self, inputs, training=False):

        x = self.convs[0](inputs)
        x = self.norms[0](x)
        for conv, norm in zip(self.convs[1:], self.norms[1:]):
            x = conv(x)
            x = norm(x, training=training)
        assert x.shape[1:] == self.last_kernel_shape
        x = self.flatten(x)

        x = self.out(x)
        return x


class Decoder(tf.keras.Model):

    def __init__(self, code_size, last_kernel_shape, output_shape, filters, kernel_sizes):
        super(Decoder, self).__init__()

        assert len(last_kernel_shape) == 2
        # (x, y) x = # of samples, y = samples n variables
        assert len(output_shape) == 2

        self.code_size = code_size
        self.last_kernel_shape = last_kernel_shape
        self.expected_output_shape = output_shape

        flat_len = last_kernel_shape[0] * last_kernel_shape[1]

        self.expand = tf.keras.layers.Dense(flat_len)
        self.reshape = tf.keras.layers.Reshape(last_kernel_shape)

        self.convs = []
        self.norms = []

        for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
            l = tf.keras.layers.Conv1DTranspose(f, k)
            b = tf.keras.layers.BatchNormalization()
            self.convs.append(l)
            self.norms.append(b)

    def call(self, inputs, training=False):
        x = self.expand(inputs)
        x = self.reshape(x)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x, training=training)
            x = conv(x)
        # assert self.expected_output_shape == x.shape[1:] # TODO fix assertion
        return x


class AutoEncoder:
    def __init__(self, **kwargs):

        input_shape = kwargs["input_shape"]
        code_size = kwargs["code_size"]
        filters = kwargs["filters"]
        kernel_sizes = kwargs["kernel_sizes"]

        if "reconstruction_loss" in kwargs:
            self.reconstruction_loss = kwargs["reconstruction_loss"]
        else:
            self.reconstruction_loss = tf.keras.losses.MeanSquaredError()

        if "codes_loss" in kwargs:
            self.codes_loss = kwargs["codes_loss"]
        else:
            self.codes_loss = tf.keras.losses.MeanSquaredError()

        if "input_distance_metric" in kwargs:
            self.input_distance_metric = kwargs["input_distance_metric"]
        else:
            self.input_distance_metric = tf.keras.losses.MeanSquaredError()
            # self.input_distance_metric = _sbd  # TODO uncomment

        if "optimizer" in kwargs:
            self.optimizer = kwargs["optimizer"]
        else:
            self.optimizer = tf.keras.optimizers.Nadam(learning_rate=0.00015)

        self.first_encoder = Encoder(
            input_shape, code_size, filters, kernel_sizes)
        self.second_encoder = Encoder(
            input_shape, code_size, filters, kernel_sizes)

        decoder_filters = list(filters[:len(filters)-1])
        decoder_filters.append(input_shape[1])
        last_kernel_shape = self.first_encoder.last_kernel_shape

        self.decoder = Decoder(code_size, last_kernel_shape, input_shape, decoder_filters,
                               kernel_sizes)

# TODO ALT: pre-calc pairwise distances (SBD)
# TODO Try with and without precalculated distances, check accuracy and performance
# TODO Try trianing encoder and decoder separately (two losses)
# TODO Try other types of convolutions?
# TODO Use SBD for reconstruction loss

@tf.function
def train_step(first_input, second_input, model, alpha=0.5):
    with tf.GradientTape() as tape:
        first_code = tf.cast(model.first_encoder(first_input, training=True), dtype=tf.float64)
        second_code = tf.cast(model.second_encoder(second_input, training=True), dtype=tf.float64)
        decodes = tf.cast(model.decoder(first_code, training=True), dtype=tf.float64)
        loss_value = (1-alpha) * model.reconstruction_loss(first_input, decodes) + alpha * abs(
            model.codes_loss(first_code, second_code) - model.input_distance_metric(first_input, second_input))
        trainables = model.first_encoder.trainable_variables + \
            model.second_encoder.trainable_variables + \
            model.decoder.trainable_variables
    gradients = tape.gradient(loss_value, trainables)
    model.optimizer.apply_gradients(zip(gradients, trainables))
    return loss_value
