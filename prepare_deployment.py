"""
Prepare model files for deployment
Copy model files to multiple locations for better accessibility
"""

import shutil
from pathlib import Path

def prepare_model_files():
    """Copy model files to deployment-friendly locations"""
    
    source_dir = Path("models/artifacts")
    
    # Files to copy
    model_files = [
        "csk_best_model_random_forest.pkl",
        "venue_encoder.pkl", 
        "opponent_encoder.pkl",
        "model_metadata.json",
        "feature_names.json"
    ]
    
    # Target directories
    target_dirs = [
        Path("."),  # Root directory
        Path("dashboards"),  # Same directory as streamlit app
        Path("models")  # Models directory
    ]
    
    print("🚀 Preparing model files for deployment...")
    
    for target_dir in target_dirs:
        target_dir.mkdir(exist_ok=True)
        
        for file_name in model_files:
            source_file = source_dir / file_name
            target_file = target_dir / file_name
            
            if source_file.exists():
                try:
                    shutil.copy2(source_file, target_file)
                    print(f"✅ Copied {file_name} to {target_dir}")
                except Exception as e:
                    print(f"❌ Failed to copy {file_name} to {target_dir}: {e}")
            else:
                print(f"⚠️ Source file not found: {source_file}")
    
    print("\n🎯 Model files prepared for deployment!")
    print("📁 Files are now available in multiple locations for better accessibility")

if __name__ == "__main__":
    prepare_model_files()
