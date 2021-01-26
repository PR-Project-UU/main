from base62 import encode
from logging import getLogger
from model import Generator, generator_loss, Discriminator, discriminator_loss
import numpy as np
from os import path
import pandas as pd
import pickle
from sys import stdout
import tensorflow as tf
from time import time

class Dataset():
    dataframe = pd.read_csv('./meta_features.csv')
    datapoints = []
    load_path: str = None
    log = getLogger('dataset')

    def __init__(self, load_path: str):
        self.load_path = load_path

        for city in self.dataframe['METROREG'].unique():
            subset = self.dataframe.loc[self.dataframe['METROREG'] == city]
            coordinates = '%.2f_%.2f' % (subset['latitude'].iloc[0], subset['longitude'].iloc[0])
            interval = [min(subset['TIME']), max(subset['TIME']) + 1]

            for year in range(interval[0], interval[1]):
                # If next year exists stop adding this city
                if not path.exists(path.join(load_path, '%s-0-%d.pickle' % (coordinates, year + 1))):
                    break

                # Get meta data (in to steps to ensure order)
                try:
                    meta_data = subset[subset['TIME'] == year][['employed_persons', 'gdp', 'population']].to_dict('records')[0]
                except IndexError:
                    self.log.error('Missing year %d data from city "%s"', year, city)
                    continue

                meta = [meta_data['employed_persons'], meta_data['gdp'], meta_data['population']]

                for i in range(25):
                    inp, tar = ('%s-%d-%d.pickle' % (coordinates, i, year), '%s-%d-%d.pickle' % (coordinates, i, year + 1))

                    if path.exists(path.join(load_path, inp)) and path.exists(path.join(load_path, tar)):
                        self.datapoints.append((inp, tar, meta))
                    else:
                        self.log.warning('Missing pair for city %s, index %d, year %d/%d', city, i, year, year + 1)

        self.log.info('Created dataset object with size %d', len(self.datapoints))

    def generator(self):
        for (inp, tar, meta) in self.datapoints:
            with open(path.join(self.load_path, inp), 'rb') as inp_file:
                inp_image = pickle.load(inp_file)
            
            with open(path.join(self.load_path, tar), 'rb') as tar_file:
                tar_image = pickle.load(tar_file)

            # meta_reshaped = tf.repeat(tf.repeat(np.moveaxis(tf.expand_dims(tf.expand_dims(meta, 1, 0), 1, 0), 0, -1), 64, axis=0), 64, axis=1)

            if np.isnan(np.sum(inp_image)):
                self.log.debug('Input image contains NaN: "%s"', inp)
                continue
            elif np.isnan(np.sum(tar_image)):
                self.log.debug('Target image contains NaN: "%s"', tar)
                continue
            # elif np.isnan(np.sum(meta)):
            #     self.log.debug('Meta data contains NaN: "%s"', inp)
            #     continue
            
            # breakpoint()

            yield (tf.cast(inp_image, tf.float32), tf.cast(tar_image, tf.float32))
            # yield (tf.cast(inp_image, tf.float32), tf.cast(tar_image, tf.float32), tf.cast(meta_reshaped, tf.float32))

    def dataset(self):
        return tf.data.Dataset.from_generator(self.generator, output_types=(tf.float32, tf.float32), output_shapes=([64, 64, 3], [64, 64, 3]))
        # return tf.data.Dataset.from_generator(self.generator, output_types=(tf.float32, tf.float32, tf.float32), output_shapes=([64, 64, 3], [64, 64, 3], [64, 64, 3]))

class Trainer():
    dataset = None
    discriminator = Discriminator()
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, 0.5)
    epochs = 150
    generator = Generator()
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, 0.5)
    log = getLogger('trainer')
    save_path = None

    def __init__(self, dataset, model: str = None, save_path: str = None, epochs: int = 150, batch_size: int = 1):
        self.dataset = dataset.dataset().shuffle(epochs * 10).batch(batch_size)
        self.epochs = epochs
        self.save_path = save_path

        self.log.debug('Setup trainer with %d epochs', epochs)

        if not model is None:
            with open(model, 'rb') as f:
                gen_weights, disc_weights = pickle.load(f)
            
            self.generator.set_weights(gen_weights)
            self.discriminator.set_weights(disc_weights)

            self.log.info('Loaded model from "%s"', model)

    def fit(self):
        ds = iter(self.dataset)

        for epoch in range(self.epochs):
            for step in range(10): # NOTE: The range here limits the steps per epoch
                input_image, target = next(ds)
                self.train_step(input_image, target)
                # input_image, target, meta = next(ds)
                # self.train_step(input_image, target, meta)
                self.log.info('Step %d', step + 1)

            # Save every 20 epochs if so desired
            if (epoch + 1) % 20 == 0 and not self.save_path is None:
                self.log.info('Reached epoch %s; saving model', epoch + 1)
                self.save()
            else:
                self.log.info('Reached epoch %s of %s', epoch + 1, self.epochs)

        if self.save_path is None:
            name = 'model-' + encode(int(time())) + '.pickle'

            self.log.info('Reached end of training, storing to "./%s"', name)
            with open(path.join('./', name, 'wb')) as f:
                pickle.dump((self.generator.get_weights(), self.discriminator.get_weights()), f)
        else:
            self.save()
            self.log.info('Reached end of training')

    def save(self):
        '''Saves the trained models' weights to file'''
        name = encode(int(time()))[2:] + '.pickle'

        with open(path.join(self.save_path, name), 'wb') as f:
            pickle.dump((self.generator.get_weights(), self.discriminator.get_weights()), f)
        
        self.log.info('Saved model to "%s"', name)

    @tf.function
    def train_step(self, input_image, target):
    # def train_step(self, input_image, target, meta):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training = True)
            # gen_output = self.generator([input_image, meta], training = True)
        
            # disc_real_output = self.discriminator([input_image, meta, target], training = True)
            # disc_gen_output = self.discriminator([input_image, meta, gen_output], training = True)
            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_gen_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, _, _ = generator_loss(disc_gen_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_gen_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))
