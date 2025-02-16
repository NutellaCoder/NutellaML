# nutella sdk 기본 기능 테스트. (init, log)

# 학습 데이터 설정
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
reshape_x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
reshape_x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 모델 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

model = Sequential()
model.add(layers.InputLayer(input_shape=(28, 28, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

# 모델 학습
history = model.fit(
    reshape_x_train,
    y_train,
    batch_size=128,
    epochs=10, #50,
    validation_split=.1
)

# 원하는 지표 생성
acc = history.history['acc']

import nutellaAgent

nnn = nutellaAgent.Nutella()
nnn.init("r1", "111", 0)
# nnn.config(batchSize=125,epoch=30)
nnn.log(accuracy=acc)


