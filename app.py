import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie

# -----------------------------------------
# 1. Page Configuration & UI Setup
# -----------------------------------------
st.set_page_config(
    page_title="OsteoAI | Knee Severity Predictor",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load Lottie animations for fluid UX
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load a medical/AI animation
lottie_medical = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_5n8oxcpw.json")

# -----------------------------------------
# 2. Model Loading (Cached for Performance)
# -----------------------------------------
@st.cache_resource
def load_cnn_model():
    # Load the trained model from Phase 2
    return tf.keras.models.load_model('knee_osteoarthritis_cnn.keras')

model = load_cnn_model()

# Define the grading scale (Kellgren-Lawrence grades)
CLASS_NAMES = ['0: Healthy', '1: Doubtful', '2: Minimal', '3: Moderate', '4: Severe']

# -----------------------------------------
# 3. Interactive Visualization Functions
# -----------------------------------------
def create_gauge_chart(probability, severity_index):
    """Creates an interactive Plotly gauge chart indicating severity."""
    # Color coding based on severity
    colors = ['#00CC96', '#AB63FA', '#FFA15A', '#FF6692', '#FF97FF']
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = severity_index,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Predicted Grade: {CLASS_NAMES[severity_index]}<br><span style='font-size:0.8em;color:gray'>Confidence: {probability*100:.2f}%</span>"},
        gauge = {
            'axis': {'range': [0, 4], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': colors[severity_index]},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.8], 'color': '#e6ffe6'},
                {'range': [0.8, 1.8], 'color': '#f2e6ff'},
                {'range': [1.8, 2.8], 'color': '#fff0e6'},
                {'range': [2.8, 3.8], 'color': '#ffe6eb'},
                {'range': [3.8, 4.0], 'color': '#ffe6ff'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': severity_index}
        }
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=300)
    return fig

# -----------------------------------------
# 4. Image Preprocessing
# -----------------------------------------
def process_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize exactly as done in training
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------------------
# 5. Dashboard Layout & Architecture
# -----------------------------------------
# Sidebar
with st.sidebar:
    st.title("🦴 OsteoAI")
    st.write("Deep Learning for Managers Project")
    st.markdown("---")
    st.write("**Methodology:**")
    st.write("Utilizes a Convolutional Neural Network (MobileNetV2) trained on X-ray imaging to predict the Kellgren-Lawrence severity grade of Knee Osteoarthritis.")
    st.markdown("---")
    st.info("Upload an X-ray to generate a diagnostic prediction.")

# Main Body
col1, col2 = st.columns([1, 2])

with col1:
    if lottie_medical:
        st_lottie(lottie_medical, height=200, key="medical_animation")
    st.markdown("### Upload X-Ray Scan")
    uploaded_file = st.file_uploader("Choose a file (JPEG/PNG)", type=['jpg', 'jpeg', 'png'])

with col2:
    st.markdown("### Diagnostic Analysis")
    if uploaded_file is None:
        st.write("Awaiting image upload to begin analysis...")
    else:
        # Display uploaded image smoothly
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Radiograph', width=300)
        
        with st.spinner("Executing Convolutional Analysis..."):
            # Process and Predict
            processed_img = process_image(image)
            predictions = model.predict(processed_img)
            
            # Extract results
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            st.markdown("---")
            # Render the Plotly Gauge Chart
            st.plotly_chart(create_gauge_chart(confidence, predicted_class_idx), use_container_width=True)
            
            # Managerial Insight Expander
            with st.expander("View Managerial Insights & Next Steps"):
                st.write(f"**Actionable Insight:** The model indicates a **{CLASS_NAMES[predicted_class_idx]}** diagnosis.")
                if predicted_class_idx <= 1:
                    st.success("Recommendation: Standard preventative care and routine checkups.")
                elif predicted_class_idx == 2:
                    st.warning("Recommendation: Monitor closely. Advise lifestyle modifications and physical therapy.")
                else:
                    st.error("Recommendation: Escalate to a specialist for potential surgical consultation or advanced pain management.")