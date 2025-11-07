#!/usr/bin/env python3
"""
Test script for the dashboard functionality
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_predictor():
    """Test the CSKPredictor class"""
    print("🔍 Testing CSKPredictor...")
    
    try:
        from src.models.predict_model import CSKPredictor
        
        # Initialize predictor
        predictor = CSKPredictor("models")
        
        # Test data
        test_match = {
            'season': 2024,
            'venue': 'MA Chidambaram Stadium, Chepauk',
            'city': 'Chennai',
            'opponent': 'Mumbai Indians',
            'toss_winner': 'Chennai Super Kings',
            'toss_decision': 'bat',
            'stage': 'league',
            'match_number': 5
        }
        
        # Test prediction
        result = predictor.get_prediction_explanation(test_match)
        
        print("✅ Predictor working!")
        print(f"📊 Prediction: {result['prediction']}")
        print(f"🎯 Confidence: {result['confidence']:.2%}")
        print(f"🏆 Win Probability: {result['win_probability']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Predictor test failed: {e}")
        return False

def test_streamlit_imports():
    """Test streamlit app imports"""
    print("\n🔍 Testing Streamlit app imports...")
    
    try:
        # Test imports
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Test our modules
        sys.path.append('src')
        from src.models.predict_model import CSKPredictor
        
        print("✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Dashboard Test Suite")
    print("=" * 40)
    
    # Test predictor
    if not test_predictor():
        return 1
    
    # Test imports
    if not test_streamlit_imports():
        return 1
    
    print("\n🎉 All tests passed!")
    print("✅ Dashboard should work correctly")
    print("\nTo launch dashboard:")
    print("  python run_dashboard.py")
    print("  or")
    print("  python -m streamlit run dashboards/streamlit_app.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
