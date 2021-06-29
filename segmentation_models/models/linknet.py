from keras_applications import get_submodules_from_kwargs

from tensorflow_addons.layers import GroupNormalization

from ._common_blocks import Conv2dNorm, Conv1x1BnReLU, Conv3x3BnReLU
from ._utils import freeze_model, filter_keras_submodules
from ..backbones.backbones_factory import Backbones

backend = None
layers = None
models = None
keras_utils = None


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------

def DecoderUpsamplingX2Block(filters, stage, normalization):
    conv_block1_name = 'decoder_stage{}a'.format(stage)
    conv_block2_name = 'decoder_stage{}b'.format(stage)
    conv_block3_name = 'decoder_stage{}c'.format(stage)
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    add_name = 'decoder_stage{}_add'.format(stage)

    channels_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip=None):
        input_filters = backend.int_shape(input_tensor)[channels_axis]
        output_filters = backend.int_shape(skip)[channels_axis] if skip is not None else filters

        x = Conv1x1BnReLU(input_filters // 4, normalization, name=conv_block1_name)(input_tensor)
        x = layers.UpSampling2D((2, 2), name=up_name)(x)
        x = Conv3x3BnReLU(input_filters // 4, normalization, name=conv_block2_name)(x)
        x = Conv1x1BnReLU(output_filters, normalization, name=conv_block3_name)(x)

        if skip is not None:
            x = layers.Add(name=add_name)([x, skip])
        return x

    return wrapper


def DecoderTransposeX2Block(filters, stage, normalization):
    conv_block1_name = 'decoder_stage{}a'.format(stage)
    transpose_name = 'decoder_stage{}b_transpose'.format(stage)
    norm_name = 'decoder_stage{}b_norm'.format(stage)
    relu_name = 'decoder_stage{}b_relu'.format(stage)
    conv_block3_name = 'decoder_stage{}c'.format(stage)
    add_name = 'decoder_stage{}_add'.format(stage)

    channels_axis = norm_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip=None):
        input_filters = backend.int_shape(input_tensor)[channels_axis]
        output_filters = backend.int_shape(skip)[channels_axis] if skip is not None else filters

        x = Conv1x1BnReLU(input_filters // 4, normalization, name=conv_block1_name)(input_tensor)
        x = layers.Conv2DTranspose(
            filters=input_filters // 4,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transpose_name,
            use_bias=not normalization,
        )(x)

        if normalization == 'batchnorm':
            x = layers.BatchNormalization(axis=norm_axis, name=norm_axis)(x)

        elif normalization == 'groupnorm':
            x = GroupNormalization(axis=norm_axis, name=norm_name)(x)

        x = layers.Activation('relu', name=relu_name)(x)
        x = Conv1x1BnReLU(output_filters, normalization, name=conv_block3_name)(x)

        if skip is not None:
            x = layers.Add(name=add_name)([x, skip])

        return x

    return wrapper


# ---------------------------------------------------------------------
#  LinkNet Decoder
# ---------------------------------------------------------------------

def build_linknet(
        backbone,
        decoder_block,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        activation='sigmoid',
        normalization='batchnorm',
):
    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, normalization, name='center_block1')(x)
        x = Conv3x3BnReLU(512, normalization, name='center_block2')(x)

    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = decoder_block(decoder_filters[i], stage=i, normalization=normalization)(x, skip)

    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform'
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, x)

    return model


# ---------------------------------------------------------------------
#  LinkNet Model
# ---------------------------------------------------------------------

def Linknet(
        backbone_name='vgg16',
        input_shape=(None, None, 3),
        classes=1,
        activation='sigmoid',
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        decoder_block_type='upsampling',
        decoder_filters=(None, None, None, None, 16),
        decoder_normalization='batchnorm',
        **kwargs
):
    """Linknet_ is a fully convolution neural network for fast image semantic segmentation

    Note:
        This implementation by default has 4 skip connections (original - 3).

    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
                    extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
                case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
                able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
            (e.g. ``sigmoid``, ``softmax``, ``linear``).
        weights: optional, path to model weights.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
                    Each of these layers will be concatenated with corresponding decoder block. If ``default`` is used
                    layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
        decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks,
            for block with skip connection a number of filters is equal to number of filters in
            corresponding encoder block (estimates automatically and can be passed as ``None`` value).
        decoder_normalization: one of 'batchnorm', 'groupnorm', and None.
        decoder_block_type: one of
                    - `upsampling`:  use ``UpSampling2D`` keras layer
                    - `transpose`:   use ``Transpose2D`` keras layer

    Returns:
        ``keras.models.Model``: **Linknet**

    .. _Linknet:
        https://arxiv.org/pdf/1707.03718.pdf
    """

    global backend, layers, models, keras_utils
    submodule_args = filter_keras_submodules(kwargs)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)

    if decoder_block_type == 'upsampling':
        decoder_block = DecoderUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        decoder_block = DecoderTransposeX2Block
    else:
        raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_block_type))

    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs,
    )

    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_linknet(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        activation=activation,
        n_upsample_blocks=len(decoder_filters),
        normalization=decoder_normalization,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model
