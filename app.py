from time import sleep
import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import random
import requests
from io import BytesIO

import tensorflow as tf
from tensorflow.keras.applications import efficientnet

def preprocess(img):
    img = efficientnet.preprocess_input(np.array(img.resize((260, 260))))
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)


def predict(image):
    with st.spinner("Computing ..."):
        interpreter.set_tensor(input_details["index"], image)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
    prediction = np.argmax(out, axis=1)[0]
    return classes[prediction]

def get_sample_images():
    dataset_path = Path("./sample")
    images = list(dataset_path.glob(r'*.png'))
    labels = [" ".join(i.stem.split("-")[:-1]) for i in images]
    imgs = {i:l for i, l in zip(images, labels)}
    return imgs

def load_image(url): 
    res = requests.get(url)
    if res.status_code == 200 and 'png' in res.headers['Content-Type']:
        print("Hello")
        img_arr = np.array(Image.open(BytesIO(res.content)))
        return img_arr
    else: 
        return None

if __name__ == "__main__":
    st.set_page_config(page_title="Weed Seedings Detection")
    TFLITE_MODEL = Path("model.tflite")

    with st.spinner("Please wait initialising the model"):
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL.__str__())
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        sample_images = get_sample_images()



    st.title("Weed Detection in Crops")

    info_md = Path("app.md").read_text()
    st.markdown(info_md)

    classes = {0: 'Black-grass', 1: 'Charlock', 2: 'Cleavers', 3: 'Common Chickweed',  5: 'Fat Hen',
               6: 'Loose Silky-bent',  8: 'Scentless Mayweed', 9: 'Shepherd\'s Purse', 10: 'Small-flowered Cranesbill',
               11: 'Sugar beet', 4: 'Common wheat', 7: 'Maize'}

    weeds = {'Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Fat Hen',
             'Loose Silky-bent',  'Scentless Mayweed', 'Shepherd\'s Purse', 'Small-flowered Cranesbill',
             'Sugar beet'}

    crops = {'Common wheat', 'Maize'}


    uploader = st.file_uploader("", type=['jpg', 'png'])


    if st.button("Use a sample image"):
        BASE = "https://raw.githubusercontent.com/atishekk/tiny_ml_w_d/main/"
        img_path = random.choice(list(sample_images.keys()))
        URI = f"{BASE}{img_path}"
        file = load_image(URI)
        if file is None:
            st.write("Error retrieving image.")
        else:
            img = Image.fromarray(file)
            img = preprocess(img)
            p = predict(img)
            st.image(
                    file, caption=f"Predicted: {p} - Correct: {sample_images[img_path]} - {'Weed' if p in weeds else 'Not a weed'}", width=500)

    elif uploader is not None:
        file = uploader.read()
        img = Image.open(uploader)
        img = preprocess(img)
        p = predict(img)
        st.image(
            file, caption=f"{p} - {'Weed' if p in weeds else 'Not a weed'}", width=500)

