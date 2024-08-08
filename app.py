import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import base64

# Set page configuration
st.set_page_config(page_title="Animal Classification", layout="wide")

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

# Replace with your image path
image_path = '/Users/mba/Downloads/animal/BG3.jpeg'
base64_image = encode_image(image_path)

# Inject custom CSS to set the background image and style elements
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_image}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        background-color: rgba(0, 0, 0, 0.8); /* Faded effect */
        color: white;
    }}
    .stHeader {{
        color: white;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.7);
    }}
    .stSubheader {{
        color: #f63366;
        font-size: 1.5rem;
    }}
    .stImage {{
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
        max-width: 100%;
        height: auto;
    }}
    .prediction-box {{
        background-color: rgba(0, 0, 0, 0.7); /* Semi-transparent box */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
        color: white;
        margin-top: 20px;
        font-size: 1.2rem; /* Increase the font size */
    }}
    .prediction-box h2 {{
        color: red;
        font-size: 2rem; /* Larger font size for heading */
    }}
    .prediction-box p {{
        font-size: 1.5rem; /* Larger font size for text */
    }}
    </style>
    """, unsafe_allow_html=True)

# Header with custom styling
st.markdown('<h1 class="stHeader">Animal Classification Model</h1>', unsafe_allow_html=True)

# Load the pre-trained model
model = load_model('/Users/mba/Downloads/animal/Image_classify.keras')

# Define the categories for classification
data_cat = ['bee', 'butterfly', 'cat', 'cow', 'crow', 'deer', 'dog', 'dolphin', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'gorilla', 'leopard', 'lion']

# Define image dimensions
img_height = 180
img_width = 180

# Upload image file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    # Load and preprocess the image
    image_load = Image.open(uploaded_file).resize((img_width, img_height))
    img_arr = np.array(image_load)  # Convert image to array
    img_bat = tf.expand_dims(img_arr, 0)  # Add batch dimension

    # Make predictions
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)

    # Display results
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    with col2:
        # Display results inside the semi-transparent box
        st.markdown(f"""
        <div class="prediction-box">
            <h2>Prediction:</h2>
            <p><strong>Animal in the image is:</strong> <em>{data_cat[np.argmax(score)]}</em></p>
            <p><strong>Confidence:</strong> <em>{np.max(score) * 100:.2f}%</em></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('---')  # Adds a horizontal line


#streamlit run app.py