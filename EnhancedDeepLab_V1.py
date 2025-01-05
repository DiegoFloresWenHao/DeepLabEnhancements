import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, padding="same", use_bias=False):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)  # Added dropout for regularization
    return x

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, num_filters=256, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, num_filters=256, kernel_size=1, dilation_rate=1)
    out_12 = convolution_block(dspp_input, num_filters=256, kernel_size=3, dilation_rate=12)
    out_24 = convolution_block(dspp_input, num_filters=256, kernel_size=3, dilation_rate=24)
    out_36 = convolution_block(dspp_input, num_filters=256, kernel_size=3, dilation_rate=36)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_12, out_24, out_36])
    output = convolution_block(x, kernel_size=1, num_filters=256)
    return output

def EnhancedDeeplab(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    
    # Use ResNet101 as backbone
    resnet101 = keras.applications.ResNet101(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    
    x = resnet101.get_layer("conv4_block23_2_relu").output  # Output of stage 4
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    
    input_b = resnet101.get_layer("conv2_block3_2_relu").output  # Low-level features from stage 2
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)
    
    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x, num_filters=256)
    x = convolution_block(x, num_filters=256)
    
    # Use transposed convolution for upsampling
    x = layers.Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(4, 4), padding='same')(x)
    
    model_output = layers.Activation('softmax')(x)  # Use softmax activation for multi-class segmentation
    
    model = tf.keras.Model(inputs=model_input, outputs=model_output)
    return model
