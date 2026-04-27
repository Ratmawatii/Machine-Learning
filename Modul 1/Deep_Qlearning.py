import tensorflow as tf
from tensorflow.keras import layers

#Bangun Deep Q-Network (DQN)
model = tf.keras.Sequential([
    layers.Dense(24, activation='relu', input_shape=(4,)),  # input: state (9 states)
    layers.Dense(24, activation='relu'),
    layers.Dense(2, activation='linear')  # output: Q-values untuk 2 aksi
])

#Loos function (Huber Loos) dan optimizer
model.compile(optimizer='adam', loss=tf.keras.losses.Huber())