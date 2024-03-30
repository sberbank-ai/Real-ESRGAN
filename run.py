import streamlit as st
from PIL import Image
import torch
import os
from RealESRGAN import RealESRGAN

# title and info
st.title("Real-ESRGAN for Super-Resolution")

image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if image_file is not None:
    lr_image = Image.open(image_file, mode='r').convert('RGB')
    st.text("Uploaded image")
    st.image(lr_image, caption='Low Resolution Image')

    scale_str = st.selectbox("Scale: ", ['x2', 'x4', 'x8'])

    pred_button = st.button("Perform Prediction")
    if pred_button:
        model = RealESRGAN(device)
        models_name = f'weights/RealESRGAN_{scale_str}.pth'
        if models_name.split('/')[-1] not in os.listdir('weights'):
            st.write("Downloading the model's weights ... (Approximate size: 68 MB)")

        model.load_weights(models_name, download=True)
        st.write("Running")

        sr_image = model.predict(lr_image)
        st.image(sr_image, caption='High Resolution Image')