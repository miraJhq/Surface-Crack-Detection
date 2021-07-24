import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
import streamlit as st
import cv2

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from glob import glob


class_names=['Positive Crack Detected ','Negative No Crack Detected']

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('model_miraJ.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

html_temp = """
    <div style="background-color:#1DB4AC ;padding:10px">
    <h2 style="color:yellow;text-align:center;">Surface Crack Detection App </h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)


file = st.file_uploader("Please upload a Crack or No Crack image in jpg/jpeg/pdf/png format only", type=["jpg","jpeg","pdf","png"])
import cv2
from PIL import Image, ImageOps
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
               
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        return prediction

if file is None:
    st.text("Please upload an jpg/jpeg/pdf/png image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    if st.button("Predict"):
      predictions = import_and_predict(image, model)
      score=np.array(predictions[0])
      st.write(score)
      st.title(
        "This Detected image most likely belongs to {} "
      .format(class_names[np.argmax(score)])
            )
        