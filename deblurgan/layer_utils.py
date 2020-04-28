import tensorflow as tf

from keras.models import Model
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, concatenate
from keras.layers.merge import Add
from keras.utils import conv_utils
# from keras.utils.conv_utils import normalize_data_format
from keras.backend.common import normalize_data_format
from keras.layers.core import Dropout


def res_block(input, filters, kernel_size=(3, 3), strides=(1, 1), use_dropout=False):
    """
    Instanciate a Keras Resnet Block using sequential API.

    :param input: Input tensor
    :param filters: Number of filters to use
    :param kernel_size: Shape of the kernel for the convolution
    :param strides: Shape of the strides for the convolution
    :param use_dropout: Boolean value to determine the use of dropout
    :return: Keras Model
    """
    x = ReflectionPadding2D((1, 1))(input)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if use_dropout:
        x = Dropout(0.5)(x)

    x = ReflectionPadding2D((1, 1))(x)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,)(x)
    x = BatchNormalization()(x)

    merged = Add()([input, x])
    return merged

def IRD_block(input, filters, kernel_size=(3, 3), strides=(1, 1),):

    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(filters=4*filters,
               kernel_size=(1, 1),
               strides=strides, padding='same')(x)

    #Tower 1
    tower_1 = Conv2D(filters=filters,
                     kernel_size=(1, 1),
                     strides=strides, padding='same')(x)
    tower_1 = BatchNormalization()(tower_1)
    tower_1 = Activation('relu')(tower_1)

    #Tower 2
    tower_2 = Conv2D(filters=4*filters,
                     kernel_size=(1, 1),
                     strides=strides, padding='same')(x)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = Activation('relu')(tower_2)

    tower_2 = Conv2D(filters=filters,
                     kernel_size=(3, 3),
                     strides=strides, padding='same')(x)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = Activation('relu')(tower_2)

    #Tower 3
    tower_3 = Conv2D(filters=filters,
                     kernel_size=(1, 1),
                     strides=strides, padding='same')(x)
    tower_3 = BatchNormalization()(tower_3)
    tower_3 = Activation('relu')(tower_3)

    tower_3 = Conv2D(filters=4*filters,
                     kernel_size=(3, 3),
                     strides=strides, padding='same')(x)
    tower_3 = BatchNormalization()(tower_3)
    tower_3 = Activation('relu')(tower_3)

    tower_3 = Conv2D(filters=filters,
                     kernel_size=(3, 3),
                     strides=strides, padding='same')(x)
    tower_3 = BatchNormalization()(tower_3)
    tower_3 = Activation('relu')(tower_3)

    #Tower 4
    # tower_4 = MaxPooling2D((2, 2), padding='same')(x)

    tower_4 = Conv2D(filters=filters,
                     kernel_size=(1, 1),
                     strides=strides, padding='same')(x)
    tower_4 = BatchNormalization()(tower_4)
    tower_4 = Activation('relu')(tower_4)

    inception = concatenate([tower_1, tower_2, tower_3, tower_4])
    inception = Add()([inception, x])

    res = BatchNormalization()(inception)
    res = Activation('relu')(x)
    res = Conv2D(filters=4*filters,
               kernel_size=(1, 1),
               strides=strides, padding='same')(x)

    output = Add()([input, res])
    return output


def spatial_reflection_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    """
    Pad the 2nd and 3rd dimensions of a 4D tensor.

    :param x: Input tensor
    :param padding: Shape of padding to use
    :param data_format: Tensorflow vs Theano convention ('channels_last', 'channels_first')
    :return: Tensorflow tensor
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))

    if data_format == 'channels_first':
        pattern = [[0, 0],
                   [0, 0],
                   list(padding[0]),
                   list(padding[1])]
    else:
        pattern = [[0, 0],
                   list(padding[0]), list(padding[1]),
                   [0, 0]]
    return tf.pad(x, pattern, "REFLECT")


# TODO: Credits
class ReflectionPadding2D(Layer):
    """Reflection-padding layer for 2D input (e.g. picture).
    This layer can add rows and columns or zeros
    at the top, bottom, left and right side of an image tensor.
    # Arguments
        padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int: the same symmetric padding
                is applied to width and height.
            - If tuple of 2 ints:
                interpreted as two different
                symmetric padding values for height and width:
                `(symmetric_height_pad, symmetric_width_pad)`.
            - If tuple of 2 tuples of 2 ints:
                interpreted as
                `((top_pad, bottom_pad), (left_pad, right_pad))`
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    # Output shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, padded_rows, padded_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, padded_rows, padded_cols)`
    """

    def __init__(self,
                 padding=(1, 1),
                 data_format=None,
                 **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                       '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    input_shape[1],
                    rows,
                    cols)
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    rows,
                    cols,
                    input_shape[3])

    def call(self, inputs):
        return spatial_reflection_2d_padding(inputs,
                                             padding=self.padding,
                                             data_format=self.data_format)

    def get_config(self):
        config = {'padding': self.padding,
                  'data_format': self.data_format}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    input = Input(shape=(256, 256, 3))
    x = ReflectionPadding2D(3)(input)
    model = Model(input, x)
    model.summary()
