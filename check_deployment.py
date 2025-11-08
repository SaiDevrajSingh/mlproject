"""
Deployment Check Script
Verify that all required files and dependencies are ready for deployment
"""

import sys
from pathlib import Path
import importlib.util

def check_requirements():
    """Check if all required packages are available"""
    print("🔍 Checking Python packages...")
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn', 
        'joblib', 'plotly', 'pathlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    return len(missing_packages) == 0

def check_model_files():
    """Check if model files are accessible"""
    print("\n📁 Checking model files...")
    
    required_files = [
        "csk_best_model_random_forest.pkl",
        "venue_encoder.pkl", 
        "opponent_encoder.pkl"
    ]
    
    search_paths = [
        Path("."),
        Path("dashboards"),
        Path("models/artifacts"),
        Path("models")
    ]
    
    files_found = {}
    
    for file_name in required_files:
        found = False
        for path in search_paths:
            file_path = path / file_name
            if file_path.exists():
                files_found[file_name] = str(file_path)
                print(f"✅ {file_name} found at {file_path}")
                found = True
                break
        
        if not found:
            print(f"❌ {file_name} - NOT FOUND")
            files_found[file_name] = None
    
    return all(path is not None for path in files_found.values()), files_found

def check_streamlit_app():
    """Check if Streamlit app can be imported"""
    print("\n🚀 Checking Streamlit app...")
    
    app_paths = [
        "dashboards/streamlit_app.py",
        "streamlit_app.py"
    ]
    
    for app_path in app_paths:
        if Path(app_path).exists():
            print(f"✅ Streamlit app found at {app_path}")
            return True
    
    print("❌ Streamlit app not found")
    return False

def test_model_loading():
    """Test if model can be loaded and used"""
    print("\n🧪 Testing model loading...")
    
    try:
        # Add dashboards to path if needed
        if Path("dashboards").exists():
            sys.path.insert(0, "dashboards")
        
        # Import and test the predictor
        from streamlit_app import CSKPredictor
        
        predictor = CSKPredictor()
        
        if not predictor._use_fallback:
            print("✅ Real ML model loaded successfully")
            
            # Test prediction
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
            print(f"✅ Test prediction successful: {result['prediction']}")
            print(f"✅ Model: {result['model_info']['model_name']}")
            return True
        else:
            print("⚠️ Using fallback predictor (model files not accessible)")
            return False
            
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def main():
    """Run all deployment checks"""
    print("🚀 CSK Prediction App - Deployment Check\n")
    
    checks = [
        ("Python Packages", check_requirements),
        ("Model Files", lambda: check_model_files()[0]),
        ("Streamlit App", check_streamlit_app),
        ("Model Loading", test_model_loading)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
            print(f"\n{'✅' if result else '❌'} {check_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results.append(False)
            print(f"\n❌ {check_name}: FAILED - {e}")
    
    print("\n" + "="*50)
    print("📊 DEPLOYMENT READINESS SUMMARY")
    print("="*50)
    
    if all(results):
        print("🎉 ALL CHECKS PASSED - Ready for deployment!")
        print("✅ Real ML model will work in deployment")
    else:
        print("⚠️ Some checks failed - Deployment will use fallback")
        print("📝 Issues to fix:")
        
        for i, (check_name, _) in enumerate(checks):
            if not results[i]:
                print(f"   - {check_name}")
    
    print("\n🚀 To deploy: Push to GitHub and deploy on Streamlit Cloud")

if __name__ == "__main__":
    main()
