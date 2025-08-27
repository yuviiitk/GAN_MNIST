# train_mnist.py
import tensorflow as tf
import numpy as np
import os
from model import build_generator, build_discriminator
from utils import save_image_grid, make_checkpoint_manager

# ---------- Hyperparams ----------
LATENT_DIM = 100
BATCH_SIZE = 128
EPOCHS = 30                  # small for test; raise for better samples
BUFFER_SIZE = 60000
SAVE_EVERY = 2               # save samples every N epochs
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Data ----------
def load_mnist_data():
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")
    # scale to [-1, 1]
    x_train = (x_train - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=-1)  # (N,28,28,1)
    return x_train

x_train = load_mnist_data()
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ---------- Models ----------
generator = build_generator(LATENT_DIM)
discriminator = build_discriminator((28,28,1))

# ---------- Losses & Opt ----------
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)

# Checkpoint manager
ckpt, manager = make_checkpoint_manager(generator, discriminator, g_optimizer, d_optimizer, ckpt_dir="checkpoints_mnist")

# ---------- Training step ----------
@tf.function
def train_step(real_images):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, LATENT_DIM])

    # Labels
    real_labels = tf.ones((batch_size, 1)) * 0.9  # label smoothing
    fake_labels = tf.zeros((batch_size, 1))

    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_images = generator(noise, training=True)

        real_preds = discriminator(real_images, training=True)
        fake_preds = discriminator(fake_images, training=True)

        d_loss_real = bce(real_labels, real_preds)
        d_loss_fake = bce(fake_labels, fake_preds)
        d_loss = d_loss_real + d_loss_fake

        # generator wants discriminator to predict ones on fake images
        g_loss = bce(tf.ones_like(fake_preds), fake_preds)

    grads_d = d_tape.gradient(d_loss, discriminator.trainable_variables)
    grads_g = g_tape.gradient(g_loss, generator.trainable_variables)

    d_optimizer.apply_gradients(zip(grads_d, discriminator.trainable_variables))
    g_optimizer.apply_gradients(zip(grads_g, generator.trainable_variables))

    return g_loss, d_loss

# ---------- Generate and save ----------
def generate_and_save(epoch, num_examples=64, seed=None):
    if seed is None:
        seed = tf.random.normal([num_examples, LATENT_DIM])
    imgs = generator(seed, training=False).numpy()  # [-1,1]
    imgs = (imgs + 1.0) / 2.0                       # [0,1]
    imgs = (imgs * 255).astype(np.uint8)
    # convert single channel to 3-channel for nicer grid handling (optional)
    imgs_rgb = np.concatenate([imgs, imgs, imgs], axis=-1)
    save_image_grid(imgs_rgb, os.path.join(OUTPUT_DIR, f"epoch_{epoch:03d}.png"), ncols=8)

# ---------- Training loop ----------
def train(epochs):
    fixed_seed = tf.random.normal([64, LATENT_DIM])  # for consistent visuals across epochs
    for epoch in range(1, epochs + 1):
        for batch in dataset:
            g_loss, d_loss = train_step(batch)

        print(f"Epoch {epoch}/{epochs} | g_loss: {g_loss:.4f} | d_loss: {d_loss:.4f}")

        if epoch % SAVE_EVERY == 0:
            generate_and_save(epoch, num_examples=64, seed=fixed_seed)
            # save checkpoint
            save_path = manager.save()
            print("Saved checkpoint:", save_path)

    # final save
    generator.save("generator_mnist.keras")
    print("Saved final generator to generator_mnist.keras")

if __name__ == "__main__":
    train(EPOCHS)
