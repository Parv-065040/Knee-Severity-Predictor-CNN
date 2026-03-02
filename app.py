import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from streamlit_lottie import st_lottie

# -----------------------------------------
# 1. Page Configuration & UI Setup
# -----------------------------------------
st.set_page_config(
    page_title="OsteoAI | Management Dashboard",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished metric cards
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        color: #31333F;
    }
    </style>
""", unsafe_allow_html=True)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_health = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_5n8oxcpw.json")

# -----------------------------------------
# 2. Model Loading (Cached)
# -----------------------------------------
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model('knee_osteoarthritis_cnn.keras')

model = load_cnn_model()
CLASS_NAMES = ['0: Healthy', '1: Doubtful', '2: Minimal', '3: Moderate', '4: Severe']

# -----------------------------------------
# 3. Interactive Visualization Functions
# -----------------------------------------
def create_gauge_chart(probability, severity_index):
    colors = ['#00CC96', '#AB63FA', '#FFA15A', '#FF6692', '#FF97FF']
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=severity_index,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Diagnostic Severity Gauge", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 4], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': colors[severity_index]},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 1.5], 'color': '#e6ffe6'},
                {'range': [1.5, 2.5], 'color': '#fff0e6'},
                {'range': [2.5, 4.0], 'color': '#ffe6eb'}],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': severity_index}
        }
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=350)
    return fig

def create_radar_chart(predictions_array):
    """Creates a radar chart to show probability distribution across all classes."""
    df = dict(
        r=predictions_array.flatten() * 100,
        theta=['Healthy', 'Doubtful', 'Minimal', 'Moderate', 'Severe']
    )
    fig = go.Figure(data=go.Scatterpolar(
        r=df['r'],
        theta=df['theta'],
        fill='toself',
        line_color='#636EFA'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title="Model Confidence Distribution (%)",
        margin=dict(l=20, r=20, t=50, b=20), height=350
    )
    return fig

# -----------------------------------------
# 4. Image Preprocessing
# -----------------------------------------
def process_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------------------
# 5. Dashboard Layout & Architecture
# -----------------------------------------
# Sidebar: Lifestyle Inputs (Connecting the disease to lifestyle factors)
with st.sidebar:
    if lottie_health:
        st_lottie(lottie_health, height=150, key="health_anim")
    st.title("🦴 OsteoAI")
    st.markdown("### Patient Lifestyle Factors")
    
    patient_age = st.slider("Patient Age", 20, 90, 55)
    patient_bmi = st.number_input("Patient BMI", 15.0, 45.0, 26.5)
    activity_level = st.selectbox("Physical Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
    
    # Simple logic to simulate a lifestyle risk multiplier
    risk_score = 0
    if patient_age > 50: risk_score += 1
    if patient_bmi > 25: risk_score += 2
    if patient_bmi > 30: risk_score += 1
    if activity_level == "Sedentary": risk_score += 2
    
    st.markdown("---")
    st.write("**System Architecture:**")
    st.caption("CNN Framework: MobileNetV2")
    st.caption("Deployment: Streamlit Cloud")

# Main Interface
st.title("Diagnostic Analytics Dashboard")
st.markdown("Assess structural joint degradation and evaluate lifestyle risk factors in a single unified view.")

# Use Tabs for a cleaner UX
tab1, tab2 = st.tabs(["🩺 Clinical Diagnosis", "📈 Model Analytics & Reporting"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Radiograph Upload")
        uploaded_file = st.file_uploader("Upload X-Ray (JPEG/PNG)", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Processed Radiograph', use_container_width=True)
            
    with col2:
        if uploaded_file is None:
            st.info("Awaiting medical image upload for automated CNN inference...")
        else:
            with st.spinner("Executing Convolutional Analysis..."):
                processed_img = process_image(image)
                predictions = model.predict(processed_img)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class_idx]
                
                # Top KPI Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric(label="Predicted Grade", value=f"Grade {predicted_class_idx}")
                m2.metric(label="AI Confidence", value=f"{confidence*100:.1f}%")
                
                # Lifestyle risk metric logic
                risk_label = "Low" if risk_score <= 2 else "Moderate" if risk_score <= 4 else "High"
                m3.metric(label="Lifestyle Risk Factor", value=risk_label, delta=f"BMI: {patient_bmi}", delta_color="inverse")
                
                st.markdown("---")
                
                # Interactive Charts Row
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(create_gauge_chart(confidence, predicted_class_idx), use_container_width=True)
                with c2:
                    st.plotly_chart(create_radar_chart(predictions[0]), use_container_width=True)

with tab2:
    st.markdown("### Managerial Insights & Clinical Action Plan")
    st.write("This section connects the deep learning output and the lifestyle inputs to concrete business and medical strategies.")
    
    if uploaded_file is not None:
        st.success(f"**Primary Finding:** The network identified a **{CLASS_NAMES[predicted_class_idx]}** structural condition.")
        
        st.markdown("#### Recommended Clinical Workflow:")
        if predicted_class_idx <= 1 and risk_score > 3:
            st.warning("Joint structure is healthy, but lifestyle metrics (BMI/Activity) indicate high risk. Recommend preventative nutrition and fitness counseling.")
        elif predicted_class_idx == 2:
            st.info("Minimal degradation detected. Recommend physical therapy and strict weight management to halt progression.")
        elif predicted_class_idx >= 3:
            st.error("Significant degradation detected. Fast-track to orthopedic specialist for surgical consultation.")
    else:
        st.write("Upload an image in the Clinical Diagnosis tab to generate the action plan.")
