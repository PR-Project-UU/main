from base62 import encode
from logging import getLogger
import numpy as np
from os import path
import pandas as pd
import pickle
from queue import SimpleQueue
from random import shuffle
import tensorflow as tf
from time import time, monotonic
from datetime import timedelta

tf.get_logger().setLevel('WARNING')

def get_discriminator():
    init = tf.keras.initializers.RandomNormal(0, 0.02)

    input_image = tf.keras.layers.Input(shape=[64, 64, 3], dtype=tf.float32)
    target_image = tf.keras.layers.Input(shape=[64, 64, 3], dtype=tf.float32)

    merged = tf.keras.layers.Concatenate()([input_image, target_image]) # (64, 64, 6)

    l = tf.keras.layers.GaussianNoise(0.2)(merged)

    l = tf.keras.layers.Conv2D(64, (4, 4), (2, 2), 'same', kernel_initializer=init)(l) # (32, 32, 64)
    l = tf.keras.layers.LeakyReLU(alpha=0.2)(l)

    l = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=init)(l) # (32, 32, 128)
    l = tf.keras.layers.BatchNormalization()(l)
    l = tf.keras.layers.LeakyReLU(alpha=0.2)(l)

    l = tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=init)(l) # (32, 32, 256)
    l = tf.keras.layers.BatchNormalization()(l)
    l = tf.keras.layers.LeakyReLU(alpha=0.2)(l)

    l = tf.keras.layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=init)(l) # (32, 32, 512)
    l = tf.keras.layers.BatchNormalization()(l)
    l = tf.keras.layers.LeakyReLU(alpha=0.2)(l)

    l = tf.keras.layers.Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(l) # (31, 31, 512)
    l = tf.keras.layers.BatchNormalization()(l)
    l = tf.keras.layers.LeakyReLU(alpha=0.2)(l)

    l = tf.keras.layers.Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(l) # (30, 30, 1)
    output = tf.keras.layers.Activation('linear')(l)

    model = tf.keras.Model([input_image, target_image], output)
    opt = tf.keras.optimizers.SGD(0.0002)
    loss = tf.keras.losses.BinaryCrossentropy(True, label_smoothing=0.1)
    model.compile(loss=loss, optimizer=opt, loss_weights=[0.5], metrics=['accuracy'])

    return model

def downsample(layer_in, filters, size=(4, 4), strides=(2, 2), apply_batchnorm = True):
    init = tf.random_normal_initializer(0., 0.02)
    l = tf.keras.layers.Conv2D(filters, size, strides, 'same', kernel_initializer=init)(layer_in)

    if apply_batchnorm:
        l = tf.keras.layers.BatchNormalization()(l, training=True)

    l = tf.keras.layers.LeakyReLU(0.2)(l)
    return l

def upsample(layer_in, skip_in, filters, size=(4, 4), strides=(2, 2), apply_dropout = False):
    init = tf.random_normal_initializer(0., 0.02)
    l = tf.keras.layers.Conv2DTranspose(filters, size, strides, 'same', kernel_initializer=init)(layer_in)
    l = tf.keras.layers.BatchNormalization()(l, training=True)

    if apply_dropout:
        l = tf.keras.layers.Dropout(0.5)(l, training=True)

    l = tf.keras.layers.Concatenate()([l, skip_in])
    l = tf.keras.layers.Activation('relu')(l)
    return l

def get_generator():
    init = tf.random_normal_initializer(0., 0.02)
    input_image = tf.keras.layers.Input(shape=[64, 64, 3])

    e1 = downsample(input_image, 64, (3, 3), (1, 1), False) # (64, 64, 64)
    e2 = downsample(e1, 128, (3, 3), (1, 1)) # (64, 64, 128)
    e3 = downsample(e2, 256) # (32, 32, 256)
    e4 = downsample(e3, 512) # (16, 16, 512)
    e5 = downsample(e4, 512) # (8, 8, 512)
    e6 = downsample(e5, 512) # (4, 4, 512)
    e7 = downsample(e6, 512) # (2, 2, 512)

    b = tf.keras.layers.Conv2D(512, (4, 4), (2, 2), 'same', kernel_initializer=init)(e7) # (1, 1, 512)
    b = tf.keras.layers.Activation('relu')(b)

    d1 = upsample(b, e7, 512, apply_dropout=True) # (2, 2, 512)
    d2 = upsample(d1, e6, 512, apply_dropout=True) # (4, 4, 512)
    d3 = upsample(d2, e5, 512, apply_dropout=True) # (8, 8, 512)
    d4 = upsample(d3, e4, 512) # (16, 16, 512)
    d5 = upsample(d4, e3, 256) # (32, 32, 256)
    d6 = upsample(d5, e2, 128) # (64, 64, 128)
    d7 = upsample(d6, e1, 64, strides=(1, 1)) # (64, 64, 64)

    output = tf.keras.layers.Conv2DTranspose(3, (4, 4), padding='same', kernel_initializer=init)(d7)
    output_image = tf.keras.layers.Activation('tanh')(output)

    model = tf.keras.Model(input_image, output_image)
    return model

def get_gan(g_model, d_model):
    d_model.trainable = False
    # for layer in d_model.layers:
    #     if not isinstance(layer, tf.keras.layers.BatchNormalization):
    #         layer.trainable = False

    input_image = tf.keras.layers.Input([64, 64, 3])
    gen_out = g_model(input_image)
    dis_out = d_model([input_image, gen_out])

    model = tf.keras.Model(input_image, [dis_out, gen_out])
    opt = tf.keras.optimizers.Adam(0.0002, 0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])

    return model

class Dataset():
    buffer = SimpleQueue()
    buffer_size = 100
    dataframe = pd.read_csv('./meta_features.csv')
    datapoints = []
    load_path: str = None
    log = getLogger('dataset')

    def __init__(self, load_path: str, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.load_path = load_path

        for city in self.dataframe['METROREG'].unique():
            subset = self.dataframe.loc[self.dataframe['METROREG'] == city]
            coordinates = '%.2f_%.2f' % (subset['latitude'].iloc[0], subset['longitude'].iloc[0])
            interval = [min(subset['TIME']), max(subset['TIME']) + 1]

            for year in range(interval[0], interval[1]):
                # If next year exists stop adding this city
                if not path.exists(path.join(load_path, '%s-0-%d.pickle' % (coordinates, year + 1))):
                    break

                for i in range(25):
                    inp, tar = ('%s-%d-%d.pickle' % (coordinates, i, year), '%s-%d-%d.pickle' % (coordinates, i, year + 1))

                    if path.exists(path.join(load_path, inp)) and path.exists(path.join(load_path, tar)):
                        self.datapoints.append((inp, tar))
                    else:
                        self.log.warning('Missing pair for city %s, index %d, year %d/%d', city, i, year, year + 1)

        self.log.info('Created dataset object with size %d', len(self.datapoints))

    def generate(self):
        nan_threshold = 122.88 # Threshold for ignoring images (1% of image content)

        while self.buffer.qsize() < self.buffer_size:
            shuffle(self.datapoints)

            for (inp, tar) in self.datapoints:
                if self.buffer.qsize() >= self.buffer_size:
                    break

                with open(path.join(self.load_path, inp), 'rb') as inp_file:
                    inp_image = pickle.load(inp_file)[None]
                
                with open(path.join(self.load_path, tar), 'rb') as tar_file:
                    tar_image = pickle.load(tar_file)[None]

                if np.isnan(inp_image).sum() < nan_threshold:
                    inp_image = np.nan_to_num(inp_image, False)
                    self.log.debug('Replaced NaNs in input image "%s"', inp)
                else:
                    self.log.debug('Input image contains NaN: "%s"', inp)
                    self.datapoints.remove((inp, tar))
                    continue

                if np.isnan(tar_image).sum() < nan_threshold:
                    tar_image = np.nan_to_num(tar_image, False)
                    self.log.debug('Replaced NaNs in target image "%s"', tar)
                else:
                    self.log.debug('Target image contains NaN: "%s"', tar)
                    self.datapoints.remove((inp, tar))
                    continue

                self.buffer.put((inp_image, tar_image))

        self.log.info('Repopulated dataset to size of %d', self.buffer_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.buffer.empty():
            self.generate()
        
        return self.buffer.get()

class Trainer:
    batch_size = 100
    batches_per_epoch = 10
    dataset = None
    discriminator = None
    load_path = None
    epochs = 150
    generator = None
    log = getLogger('trainer')
    save_path = None

    def __init__(self, load_path: str, model: str = None, save_path: str = None, epochs: int = 150, batch_size = 100, batches_per_epoch = 10):
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.dataset = Dataset(load_path, min(batch_size * batches_per_epoch, 1000))
        self.discriminator = get_discriminator()
        self.epochs = epochs
        self.generator = get_generator()
        self.load_path = load_path
        self.save_path = save_path

        if not model is None:
            with open(model, 'rb') as f:
                gen_weights, disc_weights = pickle.load(f)

            self.generator.set_weights(gen_weights)
            self.discriminator.set_weights(disc_weights)

            self.log.info('Loaded model from "%s"', model)

    def fit(self):
        start_time = monotonic()
        history = ([], [], [], [], []) # dl1, dl2, gl, da1, da2
        gan_model = get_gan(self.generator, self.discriminator)
        d_acc2 = 0.0

        for epoch in range(self.epochs):
            for batch in range(self.batches_per_epoch):
                x_inp, x_tar, x_gen, y_real, y_fake = self.generate_samples()

                # Train the generator
                if d_acc2 < 0.5:
                    d_loss1, d_acc1 = self.discriminator.train_on_batch([x_inp, x_tar], y_real)
                    d_loss2, d_acc2 = self.discriminator.train_on_batch([x_inp, x_gen], y_fake)
                else:
                   _, d_acc2 = self.discriminator.test_on_batch([x_inp, x_gen], y_fake)

                # Train the generator (via the GAN model)
                g_loss, _, _ = gan_model.train_on_batch(x_inp, [y_real, x_tar])

                history[0].append(d_loss1)
                history[1].append(d_loss2)
                history[2].append(g_loss / 10)
                history[3].append(d_acc1)
                history[4].append(d_acc2)

                self.log.info('Loss for E%3d / %3d, B%3d / %3d: d1=%.2f, d2=%.2f, da=%.2f, g=%.2f', (epoch + 1), self.epochs, (batch + 1), self.batches_per_epoch, d_loss1 or 0.0, d_loss2 or 0.0, d_acc2, g_loss)

            if (epoch + 1) % 10 == 0 and not self.save_path is None:
                self.save()

        self.save()
        self.log.info('Finished training (%d epochs) in: %s', self.epochs, timedelta(seconds = monotonic() - start_time))

        from matplotlib import pyplot as plt
        _, ax = plt.subplots(2)
        ax[0].plot(history[0], label='disc. loss real')
        ax[0].plot(history[1], label='disc. loss fake')
        ax[0].plot(history[2], label='gen. loss')
        ax[0].legend()
        ax[1].plot(history[3], label='disc. acc real')
        ax[1].plot(history[4], label='disc. acc fake')
        ax[1].legend()

        plt.show()

    def generate_samples(self):
        inp_batch = np.empty((0, 64, 64, 3), dtype='float32')
        tar_batch = np.empty((0, 64, 64, 3), dtype='float32')
        gen_batch = np.empty((0, 64, 64, 3), dtype='float32')
        
        for _ in range(self.batch_size):
            inp, tar = next(self.dataset)
            gen = self.generator(inp)

            inp_batch = np.concatenate([inp_batch, inp], 0)
            tar_batch = np.concatenate([tar_batch, tar], 0)
            gen_batch = np.concatenate([gen_batch, gen], 0)

        patch_shape = self.discriminator.output_shape[1]
        label_shape = (self.batch_size, patch_shape, patch_shape, 1)

        return inp_batch, tar_batch, gen_batch, np.ones(label_shape, 'float32'), np.zeros(label_shape, 'float32')

    def save(self):
        '''Saves the trained models' weights to file'''
        name = encode(int(time()))[2:] + '.pickle'

        with open(path.join(self.save_path, name), 'wb') as f:
            pickle.dump((self.generator.get_weights(), self.discriminator.get_weights()), f)
        
        self.log.info('Saved model to "%s"', name)