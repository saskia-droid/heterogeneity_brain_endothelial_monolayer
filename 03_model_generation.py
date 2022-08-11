import tensorflow as tf
from functools import partial

# Global Agv Pooling: features -2

def gen_model(n_classes, img_shape):

    # input
    img_input = tf.keras.Input(shape=img_shape)

    # Conv layers

    myConv = partial(tf.keras.layers.Conv2D, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                     kernel_initializer='VarianceScaling', padding='same')

    conv_1 = myConv(16, (5, 5), strides=1)(img_input)

    conv_2 = myConv(32, (3, 3), strides=2)(conv_1)
    conv_3 = myConv(16, (3, 3), strides=1)(conv_2)

    conv_4 = myConv(64, (3, 3), strides=2)(conv_3)
    conv_5 = myConv(32, (3, 3), strides=1)(conv_4)

    conv_6 = myConv(64, (3, 3), strides=2)(conv_5)
    conv_7 = myConv(32, (3, 3), strides=1)(conv_6)

    conv_8 = myConv(128, (3, 3), strides=2)(conv_7)
    conv_9 = myConv(64, (3, 3), strides=1)(conv_8)

    conv_10 = myConv(128, (3, 3), strides=2)(conv_9)
    conv_11 = myConv(64, (3, 3), strides=1)(conv_10)

    # no activation in last convolutional layer

    conv_12 = tf.keras.layers.Conv2D(64, (1, 1), strides=1, activation=None,
                                     kernel_initializer='VarianceScaling', padding='same')(conv_11)

    norm = tf.math.l2_normalize(conv_12, axis=-1)

    # Global Average pooling
    gap_features = tf.keras.layers.GlobalAveragePooling2D()(norm)

    out = tf.keras.layers.Dense(n_classes, activation='softmax')(gap_features)
    model = tf.keras.Model(img_input, out)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
