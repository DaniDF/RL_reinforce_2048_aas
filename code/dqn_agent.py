import tensorflow as tf
import numpy as np


def e_greedy(model, state, iterable_outputs, epsilon=0.1):
    if np.random.rand() < epsilon:
        result = np.random.choice(iterable_outputs)
    else:
        result = np.argmax(model(state))

    return result


class DQNAgent:
    def __init__(self, input_shape, output_space,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                 loss_fn=tf.losses.mean_squared_error):
        self.__output_space__ = output_space

        self.__model__ = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1024, input_shape=input_shape, activation="relu", dtype=tf.float32),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(len(output_space), activation="linear")
        ])
        self.__optimizer__ = optimizer
        self.__loss_fn__ = loss_fn

    def predict(self, state, policy=e_greedy, epsilon=0.1):
        return policy(self.__model__, tf.Variable([state], dtype=tf.float32), self.__output_space__, epsilon=epsilon)

    def train(self, batch, gamma=1, learning_rate=0.01, **kwargs):
        states = [item[0] for item in batch]
        next_states = [item[1] for item in batch]
        rewards = np.array([item[2] for item in batch])
        actions = np.array([item[3] for item in batch])
        dones = np.array([int(item[4]) for item in batch])
        not_dones = np.abs(np.array(dones) - 1)

        next_q1_values = self.__model__(tf.Variable(next_states, dtype=tf.float32))
        max_q1_value = np.max(next_q1_values, axis=1) * not_dones

        target_q_values = rewards + gamma * max_q1_value

        mask = tf.one_hot(actions, len(self.__output_space__))

        with tf.GradientTape() as tape:
            tape.watch(self.__model__.trainable_weights)
            q_values = self.__model__(tf.Variable(states, dtype=tf.float32))
            q_values_reduced = tf.math.reduce_sum(q_values * mask, axis=1, keepdims=True)
            # q_values_reduced = tf.math.reduce_max(q_values, axis=1, keepdims=True)

            loss = learning_rate * tf.math.reduce_mean(self.__loss_fn__(target_q_values, q_values_reduced))
            # loss = tf.math.reduce_mean(self.__loss_fn__(target_q_values, q_values_reduced))

        gradients = tape.gradient(loss, self.__model__.trainable_weights)

        self.__optimizer__.apply_gradients(zip(gradients, self.__model__.trainable_weights))

        return loss, tf.Variable([0], dtype=tf.float32), tf.Variable([0], dtype=tf.float32),\
            tf.Variable([0], dtype=tf.float32), tf.Variable([0], dtype=tf.float32), tf.Variable([0], dtype=tf.float32)

    def save_model_weights(self, filename):
        self.__model__.save_weights(filename)

    def load_model_weights(self, filename):
        self.__model__.load_weights(filename)

    def save_model(self, filename):
        self.__model__.save(filename)

    def load_model(self, filename):
        self.__model__ = tf.keras.models.load_model(filename)
