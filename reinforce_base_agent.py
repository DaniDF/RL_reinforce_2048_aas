import tensorflow as tf
import numpy as np

from threading import Lock


def sum_proba_to_one(proba):
    return proba/proba.sum()


class ReinforceBaseAgent:
    def __init__(self, input_shape, output_space,
                 optimizer_policy=tf.keras.optimizers.Adam(learning_rate=1e-3),
                 optimizer_value=tf.keras.optimizers.Adam(learning_rate=1e-3),
                 loss_fn=tf.losses.mean_squared_error):
        self.__output_space__ = output_space

        self.__model_policy__ = tf.keras.models.Sequential([
            tf.keras.layers.Dense(512, input_shape=input_shape, activation="relu", dtype=tf.float32),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(len(output_space), activation="softmax")
        ])
        self.__model_value__ = tf.keras.models.Sequential([
            tf.keras.layers.Dense(512, input_shape=input_shape, activation="relu", dtype=tf.float32),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear")
        ])

        self.__optimizer_policy__ = optimizer_policy
        self.__optimizer_value__ = optimizer_value
        self.__loss_fn__ = loss_fn
        self.__model_policy_lock__ = Lock()
        self.__model_value_lock__ = Lock()

    def predict(self, state):
        with self.__model_policy_lock__:
            linear_action_probs = self.__model_policy__(tf.Variable([state], dtype=tf.float32))[0]
        return np.random.choice(self.__output_space__, p=sum_proba_to_one(linear_action_probs.numpy()))

    def train(self, batch, alpha_value=1, alpha_policy=1):
        states = [item[0] for item in batch]
        rewards = np.array([item[2] for item in batch])
        actions = np.array([item[3] for item in batch])

        with self.__model_value_lock__:
            with tf.GradientTape() as tape_value:
                tape_value.watch(self.__model_value__.trainable_weights)

                state_value = self.__model_value__(tf.Variable(states, dtype=tf.float32))
                delta = self.__loss_fn__(rewards, state_value)
                loss_value = tf.reduce_mean(alpha_value * delta)

        with self.__model_policy_lock__:
            with tf.GradientTape() as tape_policy:
                tape_policy.watch(self.__model_policy__.trainable_weights)

                delta_policy = rewards - state_value
                delta_policy = (delta_policy - np.mean(delta_policy)) / np.sqrt(np.sum(delta_policy**2))
                delta_policy = tf.Variable(delta_policy, dtype=tf.float32)

                probs = self.__model_policy__(tf.Variable(states, dtype=tf.float32))
                mask = tf.one_hot(actions, len(self.__output_space__))
                log_prob = tf.math.log(tf.reduce_sum(probs * mask, axis=1))
                loss_policy = -tf.reduce_mean(delta_policy * log_prob * alpha_policy)

        with self.__model_value_lock__:
            gradients_value = tape_value.gradient(loss_value, self.__model_value__.trainable_weights)
            self.__optimizer_value__.apply_gradients(zip(gradients_value, self.__model_value__.trainable_weights))

        with self.__model_policy_lock__:
            gradients_policy = tape_policy.gradient(loss_policy, self.__model_policy__.trainable_weights)
            self.__optimizer_policy__.apply_gradients(zip(gradients_policy, self.__model_policy__.trainable_weights))

        return loss_policy, loss_value, tf.reduce_mean(delta_policy), tf.reduce_mean(rewards),\
            tf.reduce_mean(state_value), tf.reduce_mean(log_prob)

    def save_model_weights(self, filename):
        self.__model_policy__.save_weights(filename)

    def load_model_weights(self, filename):
        self.__model_policy__.load_weights(filename)

    def save_model(self, filename):
        self.__model_policy__.save(filename)

    def load_model(self, filename):
        self.__model_policy__ = tf.keras.models.load_model(filename)
