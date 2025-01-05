import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model

def squeeze_excitation_block(input_tensor, reduction=16):
    """Squeeze-and-Excitation block to highlight channel importance."""
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // reduction, activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, filters))(se)
    x = layers.Multiply()([input_tensor, se])
    return x

def convolution_block(
    block_input, 
    num_filters=256, 
    kernel_size=3, 
    dilation_rate=1, 
    padding="same", 
    use_bias=False, 
    attention=False,
    dropout_rate=0.0
):
    """Conv2D -> BN -> ReLU, plus optional Squeeze-Excitation and dropout."""
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal()
    )(block_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    if attention:
        x = squeeze_excitation_block(x)
    
    if dropout_rate > 0.0:
        x = Dropout(dropout_rate)(x)
        
    return x

def DilatedSpatialPyramidPooling(dspp_input, attention=False, dropout_rate=0.0):
    """ASPP module."""
    dims = dspp_input.shape
    pool_height, pool_width = dims[1], dims[2]
    
    # 1) Image Pooling
    out_pool = layers.AveragePooling2D(pool_size=(pool_height, pool_width))(dspp_input)
    out_pool = convolution_block(
        out_pool, 
        kernel_size=1, 
        use_bias=True, 
        attention=attention,
        dropout_rate=dropout_rate
    )
    # Upsample by the entire spatial dimension
    out_pool = layers.UpSampling2D(
        size=(pool_height, pool_width),
        interpolation="bilinear"
    )(out_pool)

    # 2) Parallel atrous convolutions
    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1, 
                              attention=attention, dropout_rate=dropout_rate)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6,
                              attention=attention, dropout_rate=dropout_rate)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12,
                               attention=attention, dropout_rate=dropout_rate)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18,
                               attention=attention, dropout_rate=dropout_rate)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    x = convolution_block(x, kernel_size=1, attention=attention, dropout_rate=dropout_rate)
    return x

def EnhancedDeeplab(image_size, num_classes, attention=False, dropout_rate=0.1):
    """DeepLabV3 with a fixed upsample factor for TF 2.4.1 compatibility."""
    base_model = tf.keras.applications.ResNet101(
        weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3)
    )
    # Often for ResNet101 in DeepLab:
    # 'conv4_block23_out' is stride 16
    # 'conv2_block3_out' is stride 4
    high_level_feature_name = "conv4_block23_out" 
    low_level_feature_name  = "conv2_block3_out"
    
    model_input = base_model.input
    high_level_features = base_model.get_layer(high_level_feature_name).output
    low_level_features  = base_model.get_layer(low_level_feature_name).output
    
    # ASPP on high-level features
    x = DilatedSpatialPyramidPooling(high_level_features, attention, dropout_rate)
    
    # Reduce channels on low-level features
    low_level_features = convolution_block(
        low_level_features, 
        num_filters=48, 
        kernel_size=1, 
        attention=attention, 
        dropout_rate=dropout_rate
    )
    
    # 1) Upsample from stride16 -> stride4, which is a factor of 4
    x = layers.UpSampling2D(
        size=(4, 4),  # Hardcode integer factor
        interpolation="bilinear"
    )(x)

    # 2) Concatenate
    x = layers.Concatenate(axis=-1)([x, low_level_features])
    
    # 3) Two refinement conv blocks
    x = convolution_block(x, num_filters=256, attention=attention, dropout_rate=dropout_rate)
    x = convolution_block(x, num_filters=256, attention=attention, dropout_rate=dropout_rate)
    
    # 4) Final upsample from stride4 -> stride1 (input size), factor of 4 again
    x = layers.UpSampling2D(
        size=(4, 4),  # Hardcode integer factor
        interpolation="bilinear"
    )(x)
    
    # 5) Output layer
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    
    model = Model(inputs=model_input, outputs=model_output)
    return model
