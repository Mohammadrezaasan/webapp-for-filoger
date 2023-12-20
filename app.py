import time
import cv2
import os
import streamlit as st
import pandas as pd
import numpy as np
import imageio
import base64
from PIL import Image
from tensorflow_docs.vis import embed
from tensorflow import keras
import tensorflow as tf
from moviepy.editor import VideoFileClip

# Constants
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 12
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# Load the pre-trained sequence model
sequence_model = tf.keras.models.load_model('C:\\Users\\ehsan\\Downloads\\final_model.h5')

# Build the feature extractor model
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

# Read the training dataset
train_df = pd.read_csv("C:\\Users\\ehsan\\Desktop\\exrisizes\\filoger\\20_20\\train.csv")

# Function to prepare a single video for prediction
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1

    return frame_features, frame_mask

# String lookup for labels
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
)

# Function to crop the center square of a frame
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]

# Function to load a video and preprocess its frames
def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

# Function to create a bar chart from values and labels
def chart(values, labels):
    import matplotlib.pyplot as plt

    plt.style.use('dark_background')

    plt.figure(figsize=(10, 6))

    colors = ['red', 'green', 'blue', 'yellow', 'orange']

    plt.bar(labels, values, color=colors)
    plt.xlabel('Actions', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    plt.title('Action Distribution', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.savefig("C:\\Users\\ehsan\\Desktop\\exrisizes\\filoger\\20_20\\app_photos\\chart.jpg")

# Function for sequence prediction on a video
def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("test", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]
    print(probabilities)
    print(np.argsort(probabilities)[::-1])
    list_for_chart_data = []
    values = []
    labels = []
    for i in np.argsort(probabilities)[::-1]:
        st.write(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
        list_for_chart_data.append(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    print(list_for_chart_data)
    for x in list_for_chart_data:
        x = str(x).split(':')
        labels.append(x[0])
        values.append(float(x[1].replace('%','')))
    print(labels)
    print(values)

    chart(values,labels)
    
    return frames

# Function to convert images to GIF
def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("C:\\Users\\ehsan\\Desktop\\exrisizes\\filoger\\20_20\\app_photos\\animation.gif", converted_images, fps=10)
    return embed.embed_file("C:\\Users\\ehsan\\Desktop\\exrisizes\\filoger\\20_20\\app_photos\\animation.gif")

# Function to encode a file to base64
def get_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Function to set background image for Streamlit
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''<style>.stApp {background-image: url("data:image/png;base64,%s"); background-size: cover;}</style>''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to convert a video to GIF
def video_to_gif(video_path, gif_path):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    if duration > 30:
        clip = clip.subclip(0, 30)
    clip.write_gif(gif_path, fps=10, loop=0)

def main():
    from PIL import Image

    # Set the background image
    set_background('C:\\Users\\ehsan\\Desktop\\exrisizes\\filoger\\20_20\\app_photos\\beautiful-blue-particles-bokeh-wallpaper.jpg')

    # Display an image
    image_v_2 = Image.open('C:\\Users\\ehsan\\Desktop\\exrisizes\\filoger\\20_20\\app_photos\\image.png')
    st.image(image_v_2, caption='Welcome', use_column_width=True)

    # Sidebar
    image_v_4 = Image.open('C:\\Users\\ehsan\\Desktop\\exrisizes\\filoger\\20_20\\app_photos\\circle-logo-1024x1024.png')
    st.sidebar.image(image_v_4, caption='', use_column_width=True)
    st.sidebar.markdown("""[App made for Filoger Data Science School](https://filoger.com/)""")
    st.sidebar.markdown('---')
    st.sidebar.header(("About 📑"))
    st.sidebar.markdown(("What has been coded and implemented in this project is an application software (under the web) "
                         "based on artificial neural networks and specifically convolutional and recurrent networks. "
                         "These grids with video help us identify five groups of activities."))

    st.sidebar.header(("Resources ⚙️"))
    st.sidebar.markdown(("""
    - [Our Brain](https://cdn.discordapp.com/attachments/856303332717232158/1186717105752658021/brain-stock.jpg?ex=6594436e&is=6581ce6e&hm=d7927e25c49761d8cdf9b726022939caccd2b11a865b10ad48989c88f21162a3&)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [Cheat sheet](https://docs.streamlit.io/library/cheatsheet)
    - [Chat GPT](https://openai.com/blog/chatgpt) 
    """))

    st.sidebar.header(("MMB Team 👨🏻‍💻"))
    st.sidebar.markdown(("""
    - [Mohammad Reza ASAN](https://github.com/Mohammadrezaasan)
    - [Masoud Kaviani]()
    - [Babak Heidari](https://www.linkedin.com/in/babak-heydari)
    """))

    uploaded_file = st.file_uploader("", type=["mp4", "gif", "mov", "avi"])

    if uploaded_file is not None:
        if not os.path.exists("C:\\Users\\ehsan\\Desktop\\exrisizes\\filoger\\20_20\\uploads"):
            os.makedirs("C:\\Users\\ehsan\\Desktop\\exrisizes\\filoger\\20_20\\uploads")

        # Save the uploaded video file
        video_path = os.path.join("C:\\Users\\ehsan\\Desktop\\exrisizes\\filoger\\20_20\\uploads\\", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        test_video = video_path

        # Convert the video to GIF
        video_to_gif(video_path, 'C:\\Users\\ehsan\\Desktop\\exrisizes\\filoger\\20_20\\app_photos\\animation.gif')

        with open("C:\\Users\\ehsan\\Desktop\\exrisizes\\filoger\\20_20\\app_photos\\animation.gif", "rb") as f:
            data = f.read()

        # Display the converted GIF
        st.markdown(f'<img src="data:image/gif;base64,{base64.b64encode(data).decode()}" alt="local gif" width=200>',
                    unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Prediction Results:")
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        # Perform prediction
        for percent_complete in range(88):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        test_frames = sequence_prediction(test_video)

        # Display the data chart
        path = 'C:/Users/ehsan/Desktop/exrisizes/filoger/20_20/app_photos/chart.jpg'
        image = Image.open(path)
        st.image(image, caption='Data chart', use_column_width=True)
        my_bar.progress(100, 'The operation is complete.')
        time.sleep(1)
        my_bar.empty()
        st.balloons()

if __name__ == "__main__":
    main()