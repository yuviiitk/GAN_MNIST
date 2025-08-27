# mnist_model.py
import tensorflow as tf

def build_generator(latent_dim: int = 100) -> tf.keras.Model:
    """Generator: noise (latent_dim) -> 28x28x1 image (tanh scaled)."""
    model = tf.keras.Sequential(name="generator")
    # Project and reshape
    model.add(tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Reshape((7, 7, 256)))  # 7x7x256

    # Upsample -> 7x7 -> 14x14
    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())  # 14x14x64

    # Upsample -> 28x28
    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"))
    # output: 28x28x1 in range [-1, 1]
    return model


def build_discriminator(img_shape=(28, 28, 1)) -> tf.keras.Model:
    """Discriminator: image -> probability (sigmoid)."""
    model = tf.keras.Sequential(name="discriminator")
    model.add(tf.keras.layers.Input(shape=img_shape))

    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))  # probability real/fake

    return model


if __name__ == "__main__":
    g = build_generator()
    d = build_discriminator()
    g.summary()
    d.summary()
