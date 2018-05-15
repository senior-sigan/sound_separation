# -*- coding: utf-8 -*-
# https://arxiv.org/pdf/1512.03385.pdf

import os

from keras import Input, metrics, layers
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.engine import Model, Layer
from keras.layers import Dense, K, regularizers, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, \
    Conv2D, Lambda
from keras.optimizers import Adam

from .classifier import Classifier


class ResNetClassifier(Classifier):
    def __init__(self, input_shape, labels) -> None:
        super().__init__(input_shape, labels)
        self._model = self._build()
        self._name = "ResNetClassifier"

    def _identity_layer(self, input_tensor, filters: int, kernel_size: int, strides: int,
                        stage: int, block: int) -> Layer:
        """
        Read about residual and identity block: https://arxiv.org/abs/1512.03385
        :param x:
        :param filters:
        :param kernel_size:
        :param strides:
        :return:
        """
        conv_name_base = 'res' + str(stage) + str(block) + '_branch'
        bn_name_base = 'bn' + str(stage) + str(block) + '_branch'

        with K.name_scope(name='identity_{}_{}'.format(stage, block)):
            x = Conv2D(name=conv_name_base + '2a',
                       filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=regularizers.l2(
                           l=0.0001))(input_tensor)
            x = BatchNormalization(name=bn_name_base + '2a')(x)
            x = Activation('relu')(x)

            x = Conv2D(name=conv_name_base + '2b',
                       filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=regularizers.l2(
                           l=0.0001))(x)
            x = BatchNormalization(name=bn_name_base + '2b')(x)

            # up-sample from the activation maps.
            # otherwise it's a mismatch. Recommendation of the authors.
            # here we x2 the number of filters.
            # See that as duplicating everything and concatenate them.
            if input_tensor.shape[3] != x.shape[3]:
                x = layers.add([x, Lambda(lambda y: K.repeat_elements(y, rep=2, axis=3))(input_tensor)])
            else:
                x = layers.add([x, input_tensor])
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            return x

    def _build(self):
        inputs = Input(shape=self.input_shape, dtype='float32')
        x = inputs

        x = Conv2D(filters=48,
                   kernel_size=160,
                   strides=4,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        for i in range(3):
            x = self._identity_layer(x, filters=48, kernel_size=3, strides=1, stage=1, block=i)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        for i in range(4):
            x = self._identity_layer(x, filters=96, kernel_size=3, strides=1, stage=2, block=i)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        for i in range(6):
            x = self._identity_layer(x, filters=192, kernel_size=3, strides=1, stage=3, block=i)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        for i in range(3):
            x = self._identity_layer(x, filters=384, kernel_size=3, strides=1, stage=4, block=i)
        x = GlobalAveragePooling2D()(x)

        x = Dense(len(self.labels), activation='softmax')(x)

        return Model(inputs, x, name=self.name)

    def train(self, train_gen, validation_gen, params):
        """
        :param train_gen:
        :param validation_gen:
        :param params:
            steps_per_epoch,
            epochs,
            validation_steps,
            tensorboard_dir,
            batch_size,
            chekpoints_path
        :return:
        """
        print(params)
        self._model.summary()
        self._model.compile(optimizer=Adam(),
                            loss=K.categorical_crossentropy,
                            metrics=[metrics.categorical_accuracy])

        return self._model.fit_generator(
            generator=train_gen,
            steps_per_epoch=params['steps_per_epoch'],
            epochs=params['epochs'],
            validation_data=validation_gen,
            validation_steps=params['validation_steps'],
            callbacks=[TensorBoard(
                log_dir=params['tensorboard_dir'],
                batch_size=params['batch_size']
            ), ModelCheckpoint(
                os.path.join(params['chekpoints_path'],
                             "weights-improvement-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5"),
                monitor='val_categorical_accuracy',
                verbose=1,
                save_best_only=True,
                mode='auto'
            ), ReduceLROnPlateau(
                monitor='val_acc',
                factor=0.5,
                patience=10,
                min_lr=0.0001,
                verbose=1)
            ])

    @property
    def name(self):
        return self._name
