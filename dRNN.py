import copy
import tensorflow as tf
import numpy as np

class multi_dRNN_with_dilations(tf.keras.layers.Layer):
    """
    Inputs:
        hidden_structs -- a list, each element indicates the hidden node dimension of each layer.
        dilations -- a list, each element indicates the dilation of each layer.
        input_dims -- the input dimension.
    """

    def __init__(self, cells, dilations):

        super(multi_dRNN_with_dilations, self).__init__()
        assert (len(cells) == len(dilations))
        self.cells = cells
        self.dilations =dilations

    def dRNN(self, cell, inputs, rate):
        """
        This function constructs a layer of dilated RNN.
        Inputs:
            cell -- the dilation operations is implemented independent of the RNN cell.
                In theory, any valid tensorflow rnn cell should work.
            inputs -- the input for the RNN. inputs should be in the form of
                a list of 'n_steps' tenosrs. Each has shape (batch_size, input_dims)
            rate -- the rate here refers to the 'dilations' in the orginal WaveNet paper.
            scope -- variable scope.
        Outputs:
            outputs -- the outputs from the RNN.
        """
        n_steps = len(inputs)
        if rate < 0 or rate >= n_steps:
            raise ValueError('The \'rate\' variable needs to be adjusted.')

        # make the length of inputs divide 'rate', by using zero-padding
        EVEN = (n_steps % rate) == 0
        if not EVEN:

            zero_tensor = tf.zeros_like(inputs[0])
            dialated_n_steps = n_steps // rate + 1
            for i_pad in range(dialated_n_steps * rate - n_steps):
                inputs.append(zero_tensor)
        else:
            dialated_n_steps = n_steps // rate

        dilated_inputs = [tf.concat(inputs[i * rate:(i + 1) * rate],
                                    axis=0) for i in range(dialated_n_steps)]

        # building a dialated RNN with reformated (dilated) inputs
        dilated_outputs, _ = tf.compat.v1.nn.static_rnn(cell, dilated_inputs, dtype=tf.float64)

        splitted_outputs = [tf.split(output, rate, axis=0)
                            for output in dilated_outputs]
        unrolled_outputs = [output
                            for sublist in splitted_outputs for output in sublist]
        # remove padded zeros
        outputs = unrolled_outputs[:n_steps]

        dilated_states = outputs[-1]

        return outputs , dilated_states

    def call(self, inputs):
        drnn_inputs = tf.transpose(inputs, [1, 0, 2])
        drnn_inputs = tf.unstack(drnn_inputs, axis=0)

        output = copy.copy(drnn_inputs)
        final_state_list = []
        for cell, dilation in zip(self.cells, self.dilations):
            output, state = self.dRNN(cell, output, dilation)
            final_state_list.append(state)
        output = tf.stack(output, axis=0)
        output = tf.transpose(output, [1, 0, 2])

        return output, final_state_list