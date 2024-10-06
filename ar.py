import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.image = None

    def transform(self, frame):
        # Convert the frame to a numpy array
        img = frame.to_ndarray(format="bgr")
        self.image = img  # Store the current frame
        return img

def auto_adjust_image(image):
    """Adjust the image: resize and enhance contrast."""
    # Resize the image
    height, width = image.shape[:2]
    new_width = 400  # Desired width
    aspect_ratio = width / height
    new_height = int(new_width / aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height))

    # Enhance contrast using histogram equalization
    if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:  # If image is colored
        # Convert to YUV color space
        yuv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # Equalize histogram of Y channel
        adjusted_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        # For grayscale images
        adjusted_image = cv2.equalizeHist(resized_image)

    return adjusted_image

def main():
    st.title("Breast Image Processing with Camera Capture")  # Title of the application
    
    st.write("You can either upload an image or capture a breast image from your camera.")
    
    # Image upload section
    uploaded_file = st.file_uploader("Upload a breast image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the uploaded image
        uploaded_image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        # Auto-adjust the uploaded image
        adjusted_image = auto_adjust_image(uploaded_image)
        st.image(adjusted_image, caption='Adjusted Uploaded Image', use_column_width=True)

    # Initialize the video streamer for the camera
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    # Button to capture the image
    if st.button("Capture Image"):
        if VideoTransformer.image is not None:
            # Capture the image from the video transformer
            captured_image = VideoTransformer.image.copy()
            st.image(captured_image, caption='Captured Image', use_column_width=True)

            # Auto-adjust the captured image
            adjusted_image = auto_adjust_image(captured_image)
            st.image(adjusted_image, caption='Adjusted Captured Image', use_column_width=True)

if __name__ == "__main__":
    main()
