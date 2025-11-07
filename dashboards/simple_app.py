"""
Simple CSK IPL Prediction App - Standalone version for deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date

# Page configuration
st.set_page_config(
    page_title="CSK IPL Prediction",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FFD700;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .prediction-card {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SimpleCSKPredictor:
    """Simple rule-based CSK predictor"""
    
    def __init__(self):
        self.base_win_rate = 0.56  # CSK's historical win rate
        
    def predict(self, match_data):
        """Generate prediction based on simple rules"""
        win_probability = self.base_win_rate
        key_factors = {}
        
        # Home advantage
        if self._is_home_venue(match_data.get('venue', ''), match_data.get('city', '')):
            win_probability += 0.15
            key_factors['home_advantage'] = 'Playing at home venue (Chennai/Chepauk)'
        
        # Toss advantage
        if match_data.get('toss_winner') == 'Chennai Super Kings':
            win_probability += 0.08
            key_factors['toss_advantage'] = 'Won the toss'
        
        # Peak season
        if self._is_peak_season(match_data.get('season', 2025)):
            win_probability += 0.05
            key_factors['peak_season'] = 'Peak performance season for CSK'
        
        # Strong opponent
        if self._is_strong_opponent(match_data.get('opponent', '')):
            win_probability -= 0.10
            key_factors['strong_opponent'] = f"Facing strong opponent: {match_data.get('opponent', '')}"
        
        # Playoff pressure
        if self._is_playoff_match(match_data.get('stage', '')):
            win_probability -= 0.05
            key_factors['playoff_match'] = 'High-stakes playoff match'
        
        # Ensure probability is within bounds
        win_probability = max(0.1, min(0.9, win_probability))
        
        return {
            'prediction': 'WIN' if win_probability > 0.5 else 'LOSS',
            'confidence': max(win_probability, 1 - win_probability),
            'win_probability': win_probability,
            'loss_probability': 1 - win_probability,
            'key_factors': key_factors
        }
    
    def _is_home_venue(self, venue, city):
        home_indicators = ['chennai', 'chepauk', 'ma chidambaram']
        return any(indicator in venue.lower() or indicator in city.lower() 
                  for indicator in home_indicators)
    
    def _is_peak_season(self, season):
        return season in [2010, 2011, 2018, 2021, 2023]
    
    def _is_strong_opponent(self, opponent):
        strong_opponents = ['Mumbai Indians', 'Royal Challengers Bangalore',
                          'Kolkata Knight Riders', 'Delhi Capitals']
        return opponent in strong_opponents
    
    def _is_playoff_match(self, stage):
        return stage.lower() in ['qualifier1', 'qualifier2', 'eliminator', 'final']

def main():
    # Header
    st.markdown('<h1 class="main-header">🏏 CSK IPL Performance Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predict Chennai Super Kings match outcomes using data-driven insights</p>', unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = SimpleCSKPredictor()
    
    # Sidebar inputs
    st.sidebar.header("🏏 Match Details")
    
    # Season
    season = st.sidebar.selectbox(
        "Season",
        options=list(range(2024, 2026)),
        index=0
    )
    
    # Venue
    venues = [
        "MA Chidambaram Stadium, Chepauk",
        "Wankhede Stadium",
        "Eden Gardens",
        "M Chinnaswamy Stadium",
        "Rajiv Gandhi International Stadium",
        "Sawai Mansingh Stadium",
        "Feroz Shah Kotla",
        "Punjab Cricket Association Stadium"
    ]
    venue = st.sidebar.selectbox("Venue", venues)
    
    # City
    cities = ["Chennai", "Mumbai", "Kolkata", "Bangalore", "Hyderabad", "Jaipur", "Delhi", "Mohali"]
    city = st.sidebar.selectbox("City", cities)
    
    # Opponent
    opponents = [
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
    opponent = st.sidebar.selectbox("Opponent Team", opponents)
    
    # Toss
    toss_winner = st.sidebar.selectbox(
        "Toss Winner",
        ["Chennai Super Kings", opponent, "Unknown"]
    )
    
    toss_decision = st.sidebar.selectbox(
        "Toss Decision",
        ["bat", "field"]
    )
    
    # Match stage
    stage = st.sidebar.selectbox(
        "Match Stage",
        ["league", "qualifier1", "qualifier2", "eliminator", "final"]
    )
    
    # Match number
    match_number = st.sidebar.slider("Match Number in Season", 1, 16, 8)
    
    # Prediction button
    if st.sidebar.button("🔮 Predict Match Outcome", type="primary"):
        # Prepare input data
        input_data = {
            'season': season,
            'venue': venue,
            'city': city,
            'opponent': opponent,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'stage': stage,
            'match_number': match_number
        }
        
        # Make prediction
        result = predictor.predict(input_data)
        
        # Display results
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if result['prediction'] == 'WIN':
                st.markdown(f"""
                <div class="prediction-card">
                    <h2 style="margin: 0; color: #006400;">🏆 CSK PREDICTED TO WIN!</h2>
                    <h3 style="margin: 10px 0; color: #006400;">Win Probability: {result['win_probability']:.1%}</h3>
                    <p style="margin: 0; color: #006400;">Confidence: {result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card" style="background: linear-gradient(135deg, #FF6B6B, #FF8E8E);">
                    <h2 style="margin: 0; color: #8B0000;">😔 CSK PREDICTED TO LOSE</h2>
                    <h3 style="margin: 10px 0; color: #8B0000;">Win Probability: {result['win_probability']:.1%}</h3>
                    <p style="margin: 0; color: #8B0000;">Confidence: {result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = result['win_probability'] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "CSK Win Probability (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#FFD700"},
                'steps': [
                    {'range': [0, 50], 'color': "#FF6B6B"},
                    {'range': [50, 100], 'color': "#90EE90"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key factors
        if result['key_factors']:
            st.subheader("🔍 Key Factors")
            for factor, description in result['key_factors'].items():
                st.write(f"• **{factor.replace('_', ' ').title()}**: {description}")
        
        # Match summary
        st.subheader("📊 Match Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Season", season)
            st.metric("Venue", venue)
            st.metric("Opponent", opponent)
        
        with col2:
            st.metric("Toss Winner", toss_winner)
            st.metric("Match Stage", stage.title())
            st.metric("Match Number", match_number)
    
    else:
        # Default dashboard
        st.markdown("### 🏏 Welcome to CSK Match Predictor!")
        st.write("Select match details in the sidebar and click 'Predict Match Outcome' to get predictions.")
        
        # CSK stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Historical Win Rate", "56%", "Above average")
        
        with col2:
            st.metric("IPL Titles", "4", "2010, 2011, 2018, 2021")
        
        with col3:
            st.metric("Home Advantage", "+15%", "At Chepauk")

if __name__ == "__main__":
    main()
