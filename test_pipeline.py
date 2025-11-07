#!/usr/bin/env python3
"""
Test script to run the CSK prediction pipeline step by step
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_data_extraction():
    """Test data extraction"""
    print("🔍 Testing data extraction...")
    
    try:
        from src.data.extract import DataExtractor
        
        extractor = DataExtractor('data')
        
        # Check if raw data exists
        raw_file = Path('data/raw/IPL.csv')
        if not raw_file.exists():
            print(f"❌ Raw data file not found: {raw_file}")
            return False
        
        # Load data
        df = extractor.load_ipl_data()
        print(f"✅ Data loaded: {df.shape}")
        
        # Extract CSK matches
        csk_data = extractor.extract_csk_matches(df)
        print(f"✅ CSK matches extracted: {csk_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data extraction failed: {e}")
        return False

def test_data_transformation():
    """Test data transformation"""
    print("\n🔄 Testing data transformation...")
    
    try:
        from src.data.extract import DataExtractor
        from src.data.transform import DataTransformer
        
        # Load data
        extractor = DataExtractor('data')
        df = extractor.load_ipl_data()
        csk_data = extractor.extract_csk_matches(df)
        
        # Transform data
        transformer = DataTransformer()
        cleaned_data = transformer.clean_data(csk_data)
        print(f"✅ Data cleaned: {cleaned_data.shape}")
        
        # Create CSK features
        featured_data = transformer.create_csk_features(cleaned_data)
        print(f"✅ Features created: {featured_data.shape}")
        
        # Create match-level data
        match_data = transformer.create_match_level_data(featured_data)
        print(f"✅ Match-level data: {match_data.shape}")
        
        return match_data
        
    except Exception as e:
        print(f"❌ Data transformation failed: {e}")
        return None

def test_data_loading(match_data):
    """Test data loading"""
    print("\n💾 Testing data loading...")
    
    try:
        from src.data.load import DataLoader
        
        loader = DataLoader('data')
        
        # Save processed data
        saved_path = loader.save_processed_data(
            match_data, 
            "csk_match_level_data.csv",
            {'type': 'match_level', 'ready_for_modeling': True}
        )
        print(f"✅ Data saved: {saved_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_simple_model():
    """Test simple model training"""
    print("\n🤖 Testing simple model...")
    
    try:
        from src.data.load import DataLoader
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score
        import pandas as pd
        
        # Load processed data
        loader = DataLoader('data')
        df = loader.load_processed_data("csk_match_level_data.csv")
        print(f"✅ Processed data loaded: {df.shape}")
        
        # Prepare simple features
        if 'csk_won_match' not in df.columns:
            print("❌ Target column 'csk_won_match' not found")
            return False
        
        # Select simple features
        feature_cols = []
        for col in ['venue', 'toss_winner', 'toss_decision', 'season']:
            if col in df.columns:
                feature_cols.append(col)
        
        if not feature_cols:
            print("❌ No suitable features found")
            return False
        
        X = df[feature_cols].copy()
        y = df['csk_won_match']
        
        # Simple encoding
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Train simple model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained successfully!")
        print(f"📊 Features used: {feature_cols}")
        print(f"🎯 Accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 CSK IPL Prediction - Pipeline Test")
    print("=" * 50)
    
    # Test data extraction
    if not test_data_extraction():
        return 1
    
    # Test data transformation
    match_data = test_data_transformation()
    if match_data is None:
        return 1
    
    # Test data loading
    if not test_data_loading(match_data):
        return 1
    
    # Test simple model
    if not test_simple_model():
        return 1
    
    print("\n🎉 All tests passed!")
    print("✅ Pipeline is working correctly")
    print("\nNext steps:")
    print("- Run: python run_dashboard.py")
    print("- Or use: python -m streamlit run dashboards/streamlit_app.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
