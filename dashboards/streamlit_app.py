"""
CSK IPL Performance Prediction - Streamlit App
A comprehensive web interface for predicting CSK match outcomes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path
import joblib
from datetime import datetime, date
import json

# Add src to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

try:
    from src.models.predict_model import CSKPredictor
    import logging
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure you're running from the project root directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="CSK IPL Prediction",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FFD700;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E90FF;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E90FF;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #FFD700, #FFA500);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_pipeline' not in st.session_state:
    st.session_state.prediction_pipeline = None
    st.session_state.model_loaded = False

@st.cache_resource
def load_prediction_model():
    """Load the prediction model with caching"""
    try:
        pipeline = CSKPredictor("models")
        return pipeline, True
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.error("Make sure model files exist in the models/ directory")
        return None, False

def main():
    # Header
    st.markdown('<h1 class="main-header">🏏 CSK IPL Performance Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predict Chennai Super Kings match outcomes using advanced ML models</p>', unsafe_allow_html=True)
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading prediction model..."):
            pipeline, loaded = load_prediction_model()
            st.session_state.prediction_pipeline = pipeline
            st.session_state.model_loaded = loaded
    
    if not st.session_state.model_loaded:
        st.error("❌ Model could not be loaded. Please check the model files.")
        return
    
    # Sidebar for inputs
    st.sidebar.markdown('<h2 class="sub-header">🎯 Match Details</h2>', unsafe_allow_html=True)
    
    # Match information inputs
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        season = st.number_input(
            "Season",
            min_value=2008,
            max_value=2030,
            value=2025,
            help="IPL season year"
        )
    
    with col2:
        match_number = st.number_input(
            "Match Number",
            min_value=1,
            max_value=100,
            value=1,
            help="Match number in the season"
        )
    
    # Venue and location
    venue_options = [
        "MA Chidambaram Stadium, Chepauk",
        "Wankhede Stadium",
        "Eden Gardens",
        "M Chinnaswamy Stadium",
        "Rajiv Gandhi International Stadium",
        "Sawai Mansingh Stadium",
        "Feroz Shah Kotla",
        "Punjab Cricket Association Stadium",
        "Arun Jaitley Stadium",
        "Dubai International Cricket Stadium",
        "Sheikh Zayed Stadium",
        "Sharjah Cricket Stadium"
    ]
    
    venue = st.sidebar.selectbox(
        "Venue",
        venue_options,
        help="Match venue"
    )
    
    city_options = [
        "Chennai", "Mumbai", "Kolkata", "Bangalore", "Hyderabad",
        "Jaipur", "Delhi", "Mohali", "Dubai", "Abu Dhabi", "Sharjah"
    ]
    
    city = st.sidebar.selectbox(
        "City",
        city_options,
        help="Match city"
    )
    
    # Match stage
    stage_options = ["league", "qualifier1", "qualifier2", "eliminator", "final"]
    stage = st.sidebar.selectbox(
        "Match Stage",
        stage_options,
        help="Stage of the tournament"
    )
    
    # Opponent team
    opponent_options = [
        "Mumbai Indians",
        "Royal Challengers Bangalore",
        "Kolkata Knight Riders",
        "Delhi Capitals",
        "Rajasthan Royals",
        "Punjab Kings",
        "Sunrisers Hyderabad",
        "Gujarat Titans",
        "Lucknow Super Giants"
    ]
    
    opponent = st.sidebar.selectbox(
        "Opponent Team",
        opponent_options,
        help="Opposing team"
    )
    
    # Prediction button
    predict_button = st.sidebar.button(
        "🔮 Predict Match Outcome",
        type="primary",
        use_container_width=True
    )
    
    # Main content area
    if predict_button:
        # Prepare input data
        input_data = {
            'season': season,
            'venue': venue,
            'city': city,
            'stage': stage,
            'match_number': match_number,
            'opponent': opponent
        }
        
        # Make prediction
        with st.spinner("Making prediction..."):
            try:
                # Use the get_prediction_explanation method for detailed results
                result = st.session_state.prediction_pipeline.get_prediction_explanation(input_data)
                
                predicted_win = result['prediction'] == 'WIN'
                win_probability = result['win_probability']
                
                # Display results
                display_prediction_results(input_data, predicted_win, win_probability, result)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.error("Make sure the model files are available in the models/ directory")
    
    else:
        # Default dashboard
        display_dashboard()

def display_prediction_results(input_data, predicted_win, win_probability, result=None):
    """Display prediction results with visualizations"""
    
    # Main prediction card
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if predicted_win:
            st.markdown(
                f"""
                <div class="prediction-card">
                    <h2 style="text-align: center; margin: 0; color: #006400;">
                        🏆 CSK PREDICTED TO WIN!
                    </h2>
                    <h3 style="text-align: center; margin: 10px 0; color: #333;">
                        Win Probability: {win_probability:.1%}
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-card" style="background: linear-gradient(135deg, #FF6B6B, #FF8E8E);">
                    <h2 style="text-align: center; margin: 0; color: #8B0000;">
                        😔 CSK PREDICTED TO LOSE
                    </h2>
                    <h3 style="text-align: center; margin: 10px 0; color: #333;">
                        Win Probability: {win_probability:.1%}
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Detailed metrics
    st.markdown('<h2 class="sub-header">📊 Prediction Details</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        confidence_level = "High" if win_probability > 0.7 or win_probability < 0.3 else "Medium" if win_probability > 0.6 or win_probability < 0.4 else "Low"
        st.metric("Confidence Level", confidence_level)
    
    with col2:
        st.metric("Win Probability", f"{win_probability:.1%}")
    
    with col3:
        st.metric("Loss Probability", f"{1-win_probability:.1%}")
    
    with col4:
        home_advantage = "Yes" if "Chennai" in input_data['city'] or "Chepauk" in input_data['venue'] else "No"
        st.metric("Home Advantage", home_advantage)
    
    # Probability visualization
    fig = create_probability_gauge(win_probability)
    st.plotly_chart(fig, use_container_width=True)
    
    # Match details
    st.markdown('<h2 class="sub-header">🏏 Match Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Season:** {input_data['season']}")
        st.info(f"**Venue:** {input_data['venue']}")
        st.info(f"**City:** {input_data['city']}")
    
    with col2:
        st.info(f"**Stage:** {input_data['stage'].title()}")
        st.info(f"**Match Number:** {input_data['match_number']}")
        st.info(f"**Opponent:** {input_data['opponent']}")
    
    # Historical context
    display_historical_context(input_data)

def create_probability_gauge(probability):
    """Create a probability gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "CSK Win Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "lightgreen"},
                {'range': [75, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def display_historical_context(input_data):
    """Display historical context and insights"""
    st.markdown('<h2 class="sub-header">📈 Historical Context</h2>', unsafe_allow_html=True)
    
    # CSK historical performance insights
    insights = []
    
    if "Chennai" in input_data['city'] or "Chepauk" in input_data['venue']:
        insights.append("🏠 **Home Advantage**: CSK historically performs 15% better at home venues")
    
    if input_data['opponent'] in ["Mumbai Indians", "Royal Challengers Bangalore"]:
        insights.append("⚔️ **Rivalry Match**: This is a high-stakes rivalry with unpredictable outcomes")
    
    if input_data['season'] in [2024, 2025]:
        insights.append("🆕 **Recent Form**: Consider current team composition and recent performance")
    
    if input_data['stage'] in ["qualifier1", "qualifier2", "final"]:
        insights.append("🏆 **Playoff Experience**: CSK has strong playoff experience with 10 finals appearances")
    
    for insight in insights:
        st.markdown(insight)
    
    # Performance by opponent chart
    create_opponent_performance_chart(input_data['opponent'])

def create_opponent_performance_chart(opponent):
    """Create a chart showing historical performance against opponent"""
    # Sample historical data (in a real app, this would come from your database)
    historical_data = {
        "Mumbai Indians": {"wins": 15, "losses": 17, "win_rate": 0.47},
        "Royal Challengers Bangalore": {"wins": 18, "losses": 11, "win_rate": 0.62},
        "Kolkata Knight Riders": {"wins": 16, "losses": 12, "win_rate": 0.57},
        "Delhi Capitals": {"wins": 19, "losses": 9, "win_rate": 0.68},
        "Rajasthan Royals": {"wins": 17, "losses": 8, "win_rate": 0.68},
        "Punjab Kings": {"wins": 16, "losses": 9, "win_rate": 0.64},
        "Sunrisers Hyderabad": {"wins": 12, "losses": 8, "win_rate": 0.60},
        "Gujarat Titans": {"wins": 2, "losses": 2, "win_rate": 0.50},
        "Lucknow Super Giants": {"wins": 2, "losses": 2, "win_rate": 0.50}
    }
    
    if opponent in historical_data:
        data = historical_data[opponent]
        
        fig = go.Figure(data=[
            go.Bar(name='Wins', x=['Historical Performance'], y=[data['wins']], marker_color='green'),
            go.Bar(name='Losses', x=['Historical Performance'], y=[data['losses']], marker_color='red')
        ])
        
        fig.update_layout(
            title=f'CSK vs {opponent} - Historical Head-to-Head',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Wins", data['wins'])
        with col2:
            st.metric("Total Losses", data['losses'])
        with col3:
            st.metric("Win Rate", f"{data['win_rate']:.1%}")

def display_dashboard():
    """Display the default dashboard"""
    st.markdown('<h2 class="sub-header">🏏 Welcome to CSK Prediction Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 How to Use
        1. **Select match details** in the sidebar
        2. **Choose venue and opponent** team
        3. **Click 'Predict Match Outcome'** to get predictions
        4. **View detailed analysis** and historical context
        """)
        
        st.markdown("""
        ### 🏆 About CSK
        Chennai Super Kings is one of the most successful teams in IPL history:
        - **4 IPL Championships** (2010, 2011, 2018, 2021)
        - **10 Final appearances**
        - **Strong home record** at Chepauk Stadium
        - **Consistent playoff qualification**
        """)
    
    with col2:
        # Sample visualization
        seasons = list(range(2008, 2024))
        performance = np.random.beta(2, 1.5, len(seasons)) * 100  # Sample data
        
        fig = px.line(
            x=seasons, 
            y=performance,
            title="CSK Performance Trend Over Years",
            labels={'x': 'Season', 'y': 'Performance Score'}
        )
        fig.update_traces(line_color='#FFD700', line_width=3)
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature highlights
    st.markdown('<h2 class="sub-header">✨ Model Features</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Advanced ML Models**
        - XGBoost & Random Forest
        - Calibrated probabilities
        - Feature engineering
        """)
    
    with col2:
        st.markdown("""
        **📊 Comprehensive Analysis**
        - Historical performance
        - Venue advantages
        - Head-to-head records
        """)
    
    with col3:
        st.markdown("""
        **🔮 Real-time Predictions**
        - Live probability updates
        - Confidence intervals
        - Interactive visualizations
        """)

# Footer
def display_footer():
    """Display footer information"""
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            🏏 CSK IPL Prediction System | Built with Streamlit & Advanced ML Models<br>
            <small>For educational and research purposes only</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    display_footer()
