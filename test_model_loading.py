"""
Test Model Loading
Quick test to verify model files can be loaded correctly
"""

import sys
from pathlib import Path

# Add dashboards to path
sys.path.append('dashboards')

# Import the predictor
from streamlit_app import CSKPredictor

def test_model_loading():
    """Test if model loading works"""
    print("🧪 Testing model loading...")
    
    # Test from different directories
    test_paths = [
        "models/artifacts",
        "dashboards", 
        "."
    ]
    
    for path in test_paths:
        print(f"\n📁 Testing path: {path}")
        try:
            predictor = CSKPredictor(path)
            
            if not predictor._use_fallback:
                print(f"✅ SUCCESS: Real model loaded from {path}")
                
                # Test a prediction
                test_data = {
                    'season': 2024,
                    'venue': 'MA Chidambaram Stadium, Chepauk',
                    'city': 'Chennai',
                    'opponent': 'Mumbai Indians',
                    'toss_winner': 'Chennai Super Kings',
                    'toss_decision': 'bat',
                    'stage': 'league',
                    'match_number': 8
                }
                
                result = predictor.get_prediction_explanation(test_data)
                print(f"🎯 Test prediction: {result['prediction']}")
                print(f"📊 Model: {result['model_info']['model_name']}")
                print(f"🎲 Confidence: {result['confidence']:.3f}")
                return True
            else:
                print(f"⚠️ Using fallback for path: {path}")
        except Exception as e:
            print(f"❌ Error with path {path}: {e}")
    
    print("\n❌ No real model could be loaded from any path")
    return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model loading test PASSED!")
    else:
        print("\n🚨 Model loading test FAILED!")
