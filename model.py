import tensorflow as tf

def downsample(filters, size, apply_batchnorm = True, seed = None):
    '''Creates a downsample layer group

    Args:
        filters (int): The number of filters in the Conv2D layer
        size (int): The kernal size in the Conv2D layer
        apply_batchnorm (bool, optional): Whether or not to add a batch normalization layer. Defaults to True.
        seed (int, optional): The seed for the initializer. Defaults to None.

    Returns:
        Sequential: A downsample model part
    '''
    init = tf.random_normal_initializer(0., 0.02, seed=seed)
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Conv2D(filters, size, 2, 'same', kernel_initializer=init, use_bias=False))

    if apply_batchnorm:
        model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.LeakyReLU())

    return model

def upsample(filters, size, apply_dropout = False, seed = None):
    '''Creates a upsample layer group

    Args:
        filters (int): The number of filters in the Conv2DTranspose layer
        size (int): The kernel size in the Conv2DTranspose layer
        apply_dropout (bool, optional): Whether or not to add a Dropout layer. Defaults to False.
        seed (int, optional): The seed for the initializer. Defaults to None.

    Returns:
        Sequential: A upscale model part
    '''
    init = tf.random_normal_initializer(0., 0.02, seed=seed)
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Conv2DTranspose(filters, size, 2, 'same', kernel_initializer=init, use_bias=False))

    model.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.ReLU())

    return model

def Generator():
    inputs = tf.keras.layers.Input(shape=[64, 64, 3], dtype=tf.float32)
    #meta_input = tf.keras.layers.Input(shape=[64, 64, 3], dtype=tf.float32)

    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (32, 32)
        downsample(128, 4), # (16, 16)
        downsample(256, 4), # (8, 8)
        downsample(512, 4), # (4, 4)
        downsample(512, 4), # (2, 2)
        downsample(512, 4) # (1, 1)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), # (2, 2)
        upsample(512, 4, apply_dropout=True), # (4, 4)
        upsample(512, 4, apply_dropout=True), # (8, 8)
        upsample(256, 4), # (16, 16)
        upsample(128, 4) # (32, 32)
    ]

    init = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4, 2, 'same', kernel_initializer=init, activation='tanh')

    #x = tf.keras.layers.Concatenate()([inputs, meta_input]) # Merge inputs with meta features
    x = inputs

    skips = []

    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
    #return tf.keras.Model(inputs=[inputs, meta_input], outputs=x)

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def Discriminator():
    init = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[64, 64, 3], dtype=tf.float32) # Generated image
    #meta = tf.keras.layers.Input(shape=[64, 64, 3], dtype=tf.float32) # Metadata
    tar = tf.keras.layers.Input(shape=[64, 64, 3], dtype=tf.float32) # Target image

    x = tf.keras.layers.concatenate([inp, tar]) # (64, 64, 6)
    #x = tf.keras.layers.concatenate([inp, meta, tar]) # (64, 64, 9)

    down1 = downsample(64, 4, False)(x) # (32, 32)
    down2 = downsample(128, 4)(down1) # (16, 16)
    down3 = downsample(256, 4)(down2) # (8, 8)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (10, 10)
    conv = tf.keras.layers.Conv2D(512, 4, 1, kernel_initializer=init, use_bias=False)(zero_pad1) # (7, 7)
    
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (9, 9)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=init)(zero_pad2) # (6, 6)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)
    #return tf.keras.Model(inputs=[inp, meta, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

