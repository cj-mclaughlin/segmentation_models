from keras_applications import get_submodules_from_kwargs

from tensorflow_addons.layers import GroupNormalization

def Conv2dNorm(
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        normalization='batchnorm',
        **kwargs
):
    """Extension of Conv2D layer with normalization layer (bn/gn)"""

    conv_name, act_name, norm_name = None, None, None
    block_name = kwargs.pop('name', None)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if block_name is not None:
        conv_name = block_name + '_conv'

    if block_name is not None and activation is not None:
        act_str = activation.__name__ if callable(activation) else str(activation)
        act_name = block_name + '_' + act_str

    if block_name is not None and normalization == 'batchnorm':
        norm_name = block_name + '_bn'
    
    elif block_name is not None and normalization == 'groupnorm':
        norm_name = block_name + '_gn'

    norm_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor):

        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=not normalization,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=conv_name,
        )(input_tensor)

        if normalization == 'batchnorm':
            x = layers.BatchNormalization(axis=norm_axis, name=norm_name)(x)

        elif normalization == 'groupnorm':
            x = GroupNormalization(axis=norm_axis, name=norm_name)(x)

        if activation:
            x = layers.Activation(activation, name=act_name)(x)

        return x

    return wrapper


def Conv3x3BnReLU(filters, normalization, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            normalization=normalization,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper


def Conv1x1BnReLU(filters, normalization, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dNorm(
            filters,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            normalization=normalization,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper