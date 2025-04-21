import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import cv2
from streamlit_drawable_canvas import st_canvas

# Fun title and description
st.title("🎨🖊️ **Handwritten Digit Recognizer**")
st.markdown("**Draw a digit (0-9) below and watch the magic happen! ✨**")
st.markdown("Just click **'Predict'** and we'll tell you which digit you drew. 🤔")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("/home/israa/Desktop/Deep_Learning_projects/MNIST_LSTM/mnist_lstm_model.h5")

model = load_model()

# Create a canvas for user to draw on
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Add a progress bar to make it interactive
progress_bar = st.progress(0)

if st.button("🔮 **Predict**"):
    if canvas_result.image_data is not None:
        # Update progress bar as user waits for the prediction
        progress_bar.progress(50)
        
        # Convert the canvas image to grayscale and resize to 28x28
        img = canvas_result.image_data.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = ImageOps.invert(Image.fromarray(img))
        img = np.array(img).astype("float32") / 255.0

        # Reshape to match LSTM input: (1, 28, 28)
        prediction = model.predict(img.reshape(1, 28, 28))
        predicted_class = np.argmax(prediction)

        # Complete progress bar and show the result
        progress_bar.progress(100)
        
        st.snow()
        st.success(f"🎉 **Predicted Digit:** **{predicted_class}** 🎉")
        st.markdown("![celebrate](https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif)")

        
        # Add a fun emoji or message depending on the prediction
        if predicted_class == 0:
            st.image("https://upload.wikimedia.org/wikipedia/commons/6/6f/Emoji_u0030.svg", width=50)
        elif predicted_class == 1:
            st.image("https://upload.wikimedia.org/wikipedia/commons/1/1b/Emoji_u0031.svg", width=50)
        # You can continue for all digits...


# import streamlit as st
# import numpy as np
# from PIL import Image, ImageOps
# import tensorflow as tf
# import cv2
# from streamlit_drawable_canvas import st_canvas

# st.title("🎨🖊️ **Handwritten Digit Recognizer**")
# st.markdown("**Draw a digit (0-9) below and watch the magic happen! ✨**")

# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model("/home/israa/Desktop/Deep_Learning_projects/MNIST_LSTM/mnist_lstm_model.h5")

# model = load_model()

# canvas_result = st_canvas(
#     fill_color="black",
#     stroke_width=10,
#     stroke_color="white",
#     background_color="black",
#     height=280,
#     width=280,
#     drawing_mode="freedraw",
#     key="canvas",
# )

# progress_bar = st.progress(0)

# if st.button("🔮 **Predict**"):
#     if canvas_result.image_data is not None:
#         progress_bar.progress(50)
        
#         # Convert to grayscale
#         img = canvas_result.image_data.astype(np.uint8)
#         img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

#         # Invert colors (MNIST: white digit on black)
#         img = cv2.bitwise_not(img)

#         # Resize to 28x28
#         img = cv2.resize(img, (28, 28))

#         # Normalize to [0,1]
#         img = img.astype("float32") / 255.0

#         # Reshape to (1, 28, 28)
#         img = img.reshape(1, 28, 28)

#         # Predict
#         prediction = model.predict(img)
#         predicted_class = int(np.argmax(prediction))

#         progress_bar.progress(100)
#         st.snow()
#         st.success(f"🎉 **Predicted Digit:** **{predicted_class}** 🎉")

#         if predicted_class == 0:
#             st.image("https://upload.wikimedia.org/wikipedia/commons/6/6f/Emoji_u0030.svg", width=50)

#         elif predicted_class == 1:
#             st.image("https://upload.wikimedia.org/wikipedia/commons/1/1b/Emoji_u0031.svg", width=50)
#         # You can continue for 2-9 if you want
