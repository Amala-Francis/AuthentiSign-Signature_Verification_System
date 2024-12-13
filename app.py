import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import base64

# Load the pre-trained model (replace with your model path)
model = load_model('forge_real_signature_model.keras')

# Function to preprocess the image
def preprocess_image(img, target_size=(150, 150)):
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale
    return img

# Function to encode image as base64 for CSS background
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load and encode the background image
background_image_path = 'bg4.0.jpeg'  # Path to your background image
background_base64 = image_to_base64(background_image_path)

# Custom CSS for background image
st.markdown(
    f"""
    <style>
    body {{
        background-image: url("data:image/jpeg;base64,{background_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        margin: 0;
        padding: 0;
    }}
    .stApp {{
        background: transparent;  /* Ensure no conflicting background styling here */
    }}
    .title {{
        font-family: 'Verdana, sans-serif';
        font-size: 5rem;
        color: white;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.7);
        font-style: italic;
        text-align: center;
        margin-bottom: 0.5px;
        font-weight: bold;
    }}
    .tagline {{
        font-family: 'Trebuchet MS', sans-serif;
        font-size: 1.2rem;
        color: white;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.7);
        text-align: center;
        margin-bottom: 30px;
        font-weight: bold;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.7);  /* Added shadow for the tagline */
    }}
    .mark{{
        font-family: 'Trebuchet MS', sans-serif;
        font-size: 1rem;
        text-align: center;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.7);
    }}
    .dis{{
        font-family: 'Trebuchet MS', sans-serif;
        font-size: 1rem;
        text-align: center;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.7);
    }}
    .stFileUploader {{
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.5);  /* Added shadow for file upload bars */
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(255, 255, 255, 0.7);  /* Slight background for file upload */
    }}
    
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI with Customized Title and Tagline
st.markdown("<div class='title'>AuthentiSign</div>", unsafe_allow_html=True)
st.markdown("<div class='tagline'>Your Trusted Signature Verification Solution</div>", unsafe_allow_html=True)

st.write("<div class='mark'>Upload a real signature image and a test signature image to verify if it's forged or not.</div>", unsafe_allow_html=True)

# Image upload for real signature
real_signature = st.file_uploader("Upload Real Signature", type=["png", "jpg", "jpeg"])

# Image upload for test signature
test_signature = st.file_uploader("Upload Test Signature", type=["png", "jpg", "jpeg"])

# Predict button
THRESHOLD = 0.5  # Adjust this threshold based on your model's performance

if st.button("Predict"):
    if real_signature is not None and test_signature is not None:
        # Read images from uploaded files
        real_img = image.load_img(real_signature, target_size=(150, 150))
        test_img = image.load_img(test_signature, target_size=(150, 150))

        # Convert to numpy arrays
        real_img = image.img_to_array(real_img)
        test_img = image.img_to_array(test_img)

        # Preprocess images
        real_img = preprocess_image(real_img)
        test_img = preprocess_image(test_img)

        # Get predictions for both images
        real_pred = model.predict(real_img)
        test_pred = model.predict(test_img)

        # Output predictions
        st.write(f"Real Image Prediction Confidence: {real_pred[0][0]:.2f}")
        st.write(f"Test Image Prediction Confidence: {test_pred[0][0]:.2f}")

        # Check if the test image prediction crosses the threshold
        if test_pred[0][0] > THRESHOLD:
            st.write("The signature is genuine.")
        else:
            st.write("The signature is forged.")
        
        # Display images
        st.image([real_signature, test_signature], caption=["Real Signature", "Test Signature"], use_container_width=True)
    
    else:
        st.write("Please upload both real and test signature images.")
