import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model

st.title("🧠 GAN MNIST Generator")
st.write("Generate MNIST-style digits using a trained GAN")

# Load generator model
@st.cache_resource
def load_generator():
    try:
        generator = load_model("generator.h5")  # Make sure you save generator after training
        return generator
    except:
        st.error("Generator model not found. Please train and save it as generator.h5")
        return None

generator = load_generator()

latent_dim = 100

if generator:
    st.sidebar.header("Settings")
    num_images = st.sidebar.slider("Number of images", 1, 10, 5)

    if st.button("Generate Digits"):
        noise = np.random.normal(0, 1, (num_images, latent_dim))
        generated_images = generator.predict(noise)

        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
        if num_images == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.imshow(generated_images[i, :, :, 0], cmap="gray")
            ax.axis("off")
        st.pyplot(fig)
