import streamlit as st
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np
st.title("Drowsiness detection")

st.write("Predict the person feel drowsy or not")

model = load_model("drowiness.h5")
labels = ['Closed', 'no_yawn', 'Open', 'yawn']
uploaded_file = st.file_uploader(
    "Upload an image:", type="jpg"
)
predictions=-1
if uploaded_file is not None:
    image1 = Image.open(uploaded_file)
    image1=image.smart_resize(image1,(80, 80))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = model.predict(img_array)
    predicted_label_index=np.argmax(predictions)
    if (predicted_label_index==0):
        label=('Drowsiness Detected')
    elif (predicted_label_index==3):
        label=('Drowsiness Detected')
    elif (predicted_label_index==2) and (predicted_label_index==3):
        label=('Drowsiness Detected')
    elif (predicted_label_index==0) and (predicted_label_index==1):
        label=('Drowsiness Detected')
    elif (predicted_label_index==1) & (predicted_label_index==2):
        label=('No Drowsiness Detected')
    elif (predicted_label_index==0) and (predicted_label_index==3):
        label=('Drowsiness Detected')
    elif predicted_label_index==1:
        label=('No Drowsiness Detected')
    else:
        label=("No Drowsiness Detected")



st.write("### Prediction Result")
if st.button("Predict"):
    if uploaded_file is not None:
        image1 = Image.open(uploaded_file)
        st.image(image1, caption="Uploaded Image", use_column_width=True)
        st.markdown(
            f"<h2 style='text-align: center;'>{label} in given image.</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.write("Please upload file or choose sample image.")


st.write("If you would not like to upload an image, you can use the sample image instead:")
sample_img_choice = st.button("Use Sample Image")

if sample_img_choice:
    image1 = Image.open("d2.jpeg")
    image1=image.smart_resize(image1,(80, 80))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = model.predict(img_array)
    predicted_label_index=np.argmax(predictions)
    if (predicted_label_index==0):
        label=('Drowsiness Detected')
    elif (predicted_label_index==3):
        label=('Drowsiness Detected')
    elif (predicted_label_index==2) and (predicted_label_index==3):
        label=('Drowsiness Detected')
    elif (predicted_label_index==0) and (predicted_label_index==1):
        label=('Drowsiness Detected')
    elif (predicted_label_index==1) & (predicted_label_index==2):
        label=('No Drowsiness Detected')
    elif (predicted_label_index==0) and (predicted_label_index==3):
        label=('Drowsiness Detected')
    elif predicted_label_index==1:
        label=('No Drowsiness Detected')
    else:
        label=("No Drowsiness Detected")
    image1 = Image.open("d2.jpeg")
    st.image(image1, caption="Uploaded Image", use_column_width=True)    
    st.markdown(
        f"<h2 style='text-align: center;'>{label} in given image.</h2>",
        unsafe_allow_html=True,
    )
