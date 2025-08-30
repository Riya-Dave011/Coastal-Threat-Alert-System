import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import time

# Page configuration
st.set_page_config(
    page_title="Coastal Risk Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for attractive styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Custom container styling */
    .custom-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        height: 80px;
        border-radius: 15px;
        border: none;
        font-size: 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #48cae4 0%, #023047 100%);
    }
    
    /* Metric styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(5px);
    }
    
    /* Navigation styling */
    .nav-button {
        position: fixed;
        top: 20px;
        left: 20px;
        z-index: 999;
    }
    
    /* Animation classes */
    .fade-in {
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Success/Error message styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 10px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Function to load models
@st.cache_resource
def load_models():
    """Load the trained models with progress indication"""
    tide_model = None
    coastal_model = None
    
    try:
        if os.path.exists('tide_prediction.joblib'):
            tide_model = joblib.load('tide_prediction.joblib')
        else:
            st.error("❌ tide_prediction.joblib not found")
            
        if os.path.exists('coastal_risk_model.joblib'):
            coastal_model = joblib.load('coastal_risk_model.joblib')
        else:
            st.error("❌ coastal_risk_model.joblib not found")
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
    
    return tide_model, coastal_model

# Function to make predictions
def make_predictions(tide_model, coastal_model, lat, lon):
    """Make predictions using both models"""
    results = {}
    
    # Prepare input data
    input_data = pd.DataFrame([[lat, lon]], columns=['Latitude', 'Longitude'])
    
    # Tide prediction
    if tide_model is not None:
        try:
            tide_pred = tide_model.predict(input_data)[0]
            
            # Convert to risk level
            if tide_pred >= 2.5:
                tide_level = "High"
                tide_color = "🔴"
                tide_class = "risk-high"
            elif tide_pred >= 1.5:
                tide_level = "Moderate" 
                tide_color = "🟡"
                tide_class = "risk-moderate"
            else:
                tide_level = "Low"
                tide_color = "🟢"
                tide_class = "risk-low"
                
            results['tide'] = {
                'score': tide_pred,
                'level': tide_level,
                'color': tide_color,
                'class': tide_class,
                'success': True
            }
        except Exception as e:
            results['tide'] = {'success': False, 'error': str(e)}
    
    # Coastal risk prediction
    if coastal_model is not None:
        try:
            coastal_pred = coastal_model.predict(input_data)[0]
            
            # Map to colors and classes
            if coastal_pred == "High":
                coastal_color = "🔴"
                coastal_class = "risk-high"
            elif coastal_pred == "Moderate":
                coastal_color = "🟡"
                coastal_class = "risk-moderate"
            else:
                coastal_color = "🟢"
                coastal_class = "risk-low"
                
            results['coastal'] = {
                'level': coastal_pred,
                'color': coastal_color,
                'class': coastal_class,
                'success': True
            }
        except Exception as e:
            results['coastal'] = {'success': False, 'error': str(e)}
    
    return results

# Function to display attractive prediction results
def display_prediction_results(results, latitude, longitude):
    """Display beautiful prediction results"""
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Results header with animation
    st.markdown("""
    <div class="fade-in">
        <h2 style="text-align: center; color: #fff; font-weight: 600; margin-bottom: 1rem;">
            📊 Risk Analysis Results
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Location info with attractive styling
    st.markdown(f"""
    <div class="custom-container fade-in">
        <div style="text-align: center; color: #555;">
            <strong>📍 Analysis Location:</strong> {latitude:.4f}°, {longitude:.4f}°<br>
            <strong>🕒 Generated:</strong> {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create attractive result cards
    col1, col2 = st.columns(2, gap="large")
    
    # Tide results card
    with col1:
        if 'tide' in results and results['tide']['success']:
            tide = results['tide']
            st.markdown(f"""
            <div class="result-card {tide['class']} fade-in">
                <h3 style="margin-top: 0; display: flex; align-items: center; justify-content: center;">
                    🌊 Tide Risk Analysis
                </h3>
                <div style="font-size: 2.5rem; margin: 1rem 0;">
                    {tide['color']} <strong>{tide['level']}</strong>
                </div>
                <div class="metric-container">
                    <div style="font-size: 1.8rem; font-weight: 600;">
                        {tide['score']:.2f}/3.0
                    </div>
                    <div style="opacity: 0.9;">Risk Score</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card fade-in" style="background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%);">
                <h3 style="margin-top: 0;">🌊 Tide Risk Analysis</h3>
                <div style="font-size: 1.2rem;">❌ Model Unavailable</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Coastal results card
    with col2:
        if 'coastal' in results and results['coastal']['success']:
            coastal = results['coastal']
            st.markdown(f"""
            <div class="result-card {coastal['class']} fade-in">
                <h3 style="margin-top: 0; display: flex; align-items: center; justify-content: center;">
                    🏖️ Coastal Flood Risk
                </h3>
                <div style="font-size: 2.5rem; margin: 1rem 0;">
                    {coastal['color']} <strong>{coastal['level']}</strong>
                </div>
                <div class="metric-container">
                    <div style="font-size: 1.2rem; font-weight: 600;">
                        Risk Category
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card fade-in" style="background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%);">
                <h3 style="margin-top: 0;">🏖️ Coastal Flood Risk</h3>
                <div style="font-size: 1.2rem;">❌ Model Unavailable</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Overall assessment with attractive styling
    if ('tide' in results and results['tide']['success'] and 
        'coastal' in results and results['coastal']['success']):
        
        tide_risk = results['tide']['score']
        coastal_risk = results['coastal']['level']
        
        # Determine overall risk and styling
        if tide_risk >= 2.5 or coastal_risk == "High":
            overall_class = "risk-high"
            overall_text = "🔴 CRITICAL RISK"
            overall_desc = "Avoid all coastal activities immediately"
        elif tide_risk >= 1.5 or coastal_risk == "Moderate":
            overall_class = "risk-moderate"
            overall_text = "🟡 MODERATE RISK"
            overall_desc = "Exercise extreme caution near coastal areas"
        else:
            overall_class = "risk-low"
            overall_text = "🟢 LOW RISK"
            overall_desc = "Generally safe conditions for coastal activities"
        
        st.markdown(f"""
        <div class="result-card {overall_class} fade-in pulse" style="margin-top: 2rem;">
            <h3 style="margin-top: 0;">🎯 Overall Risk Assessment</h3>
            <div style="font-size: 2rem; font-weight: 700; margin: 1rem 0;">
                {overall_text}
            </div>
            <div style="font-size: 1.1rem; opacity: 0.95;">
                {overall_desc}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'latitude' not in st.session_state:
    st.session_state.latitude = 53.3498
if 'longitude' not in st.session_state:
    st.session_state.longitude = -6.2603

# Load models
tide_model, coastal_model = load_models()

# Check if models are loaded
if tide_model is None and coastal_model is None:
    st.markdown("""
    <div class="custom-container">
        <h2 style="color: #e74c3c; text-align: center;">⚠️ Model Files Missing</h2>
        <p style="text-align: center; font-size: 1.1rem;">
            Please ensure the following files are in your app directory:
        </p>
        <ul style="text-align: center; list-style: none; font-size: 1.1rem;">
            <li>📁 tide_prediction.joblib</li>
            <li>📁 coastal_risk_model.joblib</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# HOME PAGE - Attractive landing page
if st.session_state.page == 'home':
    
    # Main title with gradient
    st.markdown("""
    <div class="fade-in">
        <h1 class="main-title">🌊 Coastal Risk Predictor</h1>
        <p class="subtitle">Advanced AI-powered coastal risk assessment for safer maritime activities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown("""
    <div class="custom-container fade-in">
        <div style="text-align: center; margin-bottom: 2rem;">
            <h3 style="color: #333; margin-bottom: 1rem;">🎯 Choose Your Analysis Method</h3>
            <p style="color: #666; font-size: 1.1rem;">Select how you'd like to specify the location for comprehensive risk analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create attractive button layout
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Map button with attractive styling
        # st.markdown("""
        # <div class="custom-container fade-in">
        # """, unsafe_allow_html=True)
        
        if st.button("🗺️ Interactive Map Selection", type="primary", use_container_width=True):
            st.session_state.page = 'map'
            st.rerun()
        
        st.markdown("""
            <div style="text-align: center; color: #666; margin-top: 0.5rem; font-size: 0.9rem;">
                Click anywhere on our interactive map to instantly select coordinates
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # st.markdown("<br>", unsafe_allow_html=True)
        
        # Coordinates button
        # st.markdown("""
        # <div class="custom-container fade-in">
        # """, unsafe_allow_html=True)
        
        if st.button("📍 Manual Coordinate Entry", type="secondary", use_container_width=True):
            st.session_state.page = 'coordinates'
            st.rerun()
        
        st.markdown("""
            <div style="text-align: center; color: #666; margin-top: 0.5rem; font-size: 0.9rem;">
                Enter precise latitude and longitude values manually
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature highlights
    
    # Features section
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
        <div class="custom-container fade-in">
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🤖</div>
                <h4 style="color: #333; margin-bottom: 0.5rem;">AI-Powered</h4>
                <p style="color: #666; font-size: 0.9rem;">Advanced machine learning models for accurate predictions</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div class="custom-container fade-in">
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">⚡</div>
                <h4 style="color: #333; margin-bottom: 0.5rem;">Real-Time</h4>
                <p style="color: #666; font-size: 0.9rem;">Instant risk assessment at any coastal location</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
        <div class="custom-container fade-in">
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🛡️</div>
                <h4 style="color: #333; margin-bottom: 0.5rem;">Safety First</h4>
                <p style="color: #666; font-size: 0.9rem;">Comprehensive risk analysis for informed decisions</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# MAP PAGE - Enhanced interactive experience
elif st.session_state.page == 'map':
    
    # Navigation
    if st.button("← Back to Home", key="nav_home"):
        st.session_state.page = 'home'
        st.rerun()
    
    st.markdown("""
    <div class="fade-in">
        <h1 style="text-align: center; color: #fff; font-weight: 600; margin-bottom: 0.5rem;">
            🗺️ Interactive Map Analysis
        </h1>
        <p style="text-align: center; color: #666; font-size: 2.2rem; margin-bottom: 2rem;">
            Click anywhere on the map to instantly analyze coastal risks at that location
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Try to use streamlit-folium for interactive map
    try:
        import folium
        from streamlit_folium import st_folium
        
        # Create beautiful folium map
        m = folium.Map(
            location=[st.session_state.latitude, st.session_state.longitude], 
            zoom_start=6,
            tiles='CartoDB positron'  # Clean, modern tile style
        )
        
        # Add attractive marker
        folium.Marker(
            [st.session_state.latitude, st.session_state.longitude],
            popup=folium.Popup(f"""
            <div style="text-align: center; font-family: Arial; min-width: 150px;">
                <h4 style="color: #2c3e50; margin-bottom: 10px;">📍 Selected Location</h4>
                <p style="margin: 5px 0;"><strong>Latitude:</strong> {st.session_state.latitude:.4f}°</p>
                <p style="margin: 5px 0;"><strong>Longitude:</strong> {st.session_state.longitude:.4f}°</p>
            </div>
            """, max_width=200),
            tooltip="Click for coordinates",
            icon=folium.Icon(color='red', icon='map-pin', prefix='fa')
        ).add_to(m)
        
        # Display interactive map in container
        # st.markdown('<div class="custom-container fade-in">', unsafe_allow_html=True)
        map_data = st_folium(m, width=None, height=500, returned_objects=["last_clicked"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle map clicks - just update coordinates
        if map_data['last_clicked']:
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lng = map_data['last_clicked']['lng']
            if clicked_lat and clicked_lng:
                st.session_state.latitude = clicked_lat
                st.session_state.longitude = clicked_lng
                st.success(f"📍 Location selected: {clicked_lat:.4f}°, {clicked_lng:.4f}°")
                st.rerun()
        
        # Manual prediction button
        # st.markdown('<div class="custom-container fade-in">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔮 Analyze Coastal Risks", type="primary", use_container_width=True):
                # Create prediction with progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔍 Initializing analysis...")
                for i in range(30):
                    progress_bar.progress(i + 1)
                    time.sleep(0.02)
                
                status_text.text("🌊 Processing tide data...")
                for i in range(30, 70):
                    progress_bar.progress(i + 1)
                    time.sleep(0.02)
                
                status_text.text("🏖️ Analyzing flood risks...")
                for i in range(70, 100):
                    progress_bar.progress(i + 1)
                    time.sleep(0.02)
                
                results = make_predictions(tide_model, coastal_model, st.session_state.latitude, st.session_state.longitude)
                
                progress_bar.empty()
                status_text.empty()
                
                st.balloons()
                display_prediction_results(results, st.session_state.latitude, st.session_state.longitude)
        st.markdown('</div>', unsafe_allow_html=True)
    
    except ImportError:
        st.markdown("""
        <div class="custom-container">
            <div style="text-align: center; color: #e74c3c;">
                <h4>📦 Enhanced Map Feature Unavailable</h4>
                <p>Install streamlit-folium for full interactive experience:</p>
                <code>pip install streamlit-folium</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Fallback map
        map_data = pd.DataFrame({
            'lat': [st.session_state.latitude],
            'lon': [st.session_state.longitude]
        })
        st.map(map_data, zoom=6)

# COORDINATES PAGE - Enhanced manual input
elif st.session_state.page == 'coordinates':
    
    # Navigation
    if st.button("← Back to Home", key="nav_home_coord"):
        st.session_state.page = 'home'
        st.rerun()
    
    st.markdown("""
    <div class="fade-in">
        <h1 style="text-align: center; color: #333; font-weight: 600; margin-bottom: 0.5rem;">
             Coordinate-Based Analysis
        </h1>
        <p style="text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 2rem;">
            Enter precise latitude and longitude coordinates for detailed risk assessment
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Coordinate inputs in attractive container
    # st.markdown('<div class="custom-container fade-in">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        latitude = st.number_input(
            "🌐 Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=st.session_state.latitude,
            step=0.0001,
            format="%.4f",
            help="Enter latitude between -90 and 90 degrees"
        )
    
    with col2:
        longitude = st.number_input(
            "🌐 Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=st.session_state.longitude,
            step=0.0001,
            format="%.4f",
            help="Enter longitude between -180 and 180 degrees"
        )
    
    # st.markdown('</div>', unsafe_allow_html=True)
    
    # Update session state
    st.session_state.latitude = latitude
    st.session_state.longitude = longitude
    
    # # Quick location presets
    # st.markdown("""
    # <div class="custom-container fade-in">
    #     <h4 style="text-align: center; color: #333; margin-bottom: 1rem;">🌍 Quick Location Presets</h4>
    # </div>
    # """, unsafe_allow_html=True)
    
    # preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    
    # presets = [
    #     ("🇮🇪 Dublin", (53.3498, -6.2603)),
    #     ("🇺🇸 Miami", (25.7617, -80.1918)),
    #     ("🇬🇧 Brighton", (50.8225, -0.1372)),
    #     ("🇦🇺 Sydney", (-33.8688, 151.2093))
    # ]
    
    # for i, (name, coords) in enumerate(presets):
    #     with [preset_col1, preset_col2, preset_col3, preset_col4][i]:
    #         if st.button(name, use_container_width=True):
    #             st.session_state.latitude, st.session_state.longitude = coords
    #             st.rerun()
    
    # # Location preview
    # st.markdown("""
    # <div class="custom-container fade-in">
    #     <h4 style="text-align: center; color: #333; margin-bottom: 1rem;">🗺️ Location Preview</h4>
    # </div>
    # """, unsafe_allow_html=True)
    
    # preview_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
    # st.map(preview_data, zoom=8)
    
    # Predict button with enhanced styling
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔮 Analyze Coastal Risks", type="primary", use_container_width=True):
            if -90 <= latitude <= 90 and -180 <= longitude <= 180:
                
                # Enhanced loading animation
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔍 Initializing analysis...")
                for i in range(30):
                    progress_bar.progress(i + 1)
                    time.sleep(0.02)
                
                status_text.text("🌊 Processing tide data...")
                for i in range(30, 70):
                    progress_bar.progress(i + 1)
                    time.sleep(0.02)
                
                status_text.text("🏖️ Analyzing flood risks...")
                for i in range(70, 100):
                    progress_bar.progress(i + 1)
                    time.sleep(0.02)
                
                results = make_predictions(tide_model, coastal_model, latitude, longitude)
                
                progress_bar.empty()
                status_text.empty()
                
                st.balloons()  # Celebration animation
                display_prediction_results(results, latitude, longitude)
            else:
                st.error("❌ Please enter valid coordinates!")

# Enhanced footer
st.markdown("""
<div class="custom-container fade-in" style="text-align: center; margin-top: 3rem;">
    <div style="color: #666; font-size: 0.9rem;">
        🌊 <strong>Coastal Risk Predictor v2.0</strong> - Powered by Advanced Machine Learning<br>
        <em style="font-size: 0.8rem;">⚠️ For informational purposes only. Always consult official weather services for safety decisions.</em>
    </div>
</div>
""", unsafe_allow_html=True)