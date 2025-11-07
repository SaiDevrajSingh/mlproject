"""
Data Quality Tests
Automated tests for data quality validation
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.extract import DataExtractor
from data.transform import DataTransformer
from data.load import DataLoader

class TestDataQuality(unittest.TestCase):
    """Test data quality at various pipeline stages"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_dir = "data"
        self.extractor = DataExtractor(self.data_dir)
        self.transformer = DataTransformer()
        self.loader = DataLoader(self.data_dir)
    
    def test_raw_data_exists(self):
        """Test that raw data file exists"""
        raw_file = Path(self.data_dir) / "raw" / "IPL.csv"
        self.assertTrue(raw_file.exists(), "Raw IPL data file should exist")
    
    def test_raw_data_schema(self):
        """Test raw data has expected schema"""
        try:
            df = self.extractor.load_ipl_data()
            
            # Check required columns exist
            required_columns = [
                'match_id', 'date', 'batting_team', 'bowling_team',
                'venue', 'toss_winner', 'toss_decision', 'match_won_by'
            ]
            
            for col in required_columns:
                self.assertIn(col, df.columns, f"Column '{col}' should exist in raw data")
            
            # Check data types
            self.assertTrue(len(df) > 0, "Raw data should not be empty")
            
        except FileNotFoundError:
            self.skipTest("Raw data file not found")
    
    def test_csk_data_extraction(self):
        """Test CSK data extraction"""
        try:
            df = self.extractor.load_ipl_data()
            csk_data = self.extractor.extract_csk_matches(df)
            
            # Check CSK matches exist
            self.assertTrue(len(csk_data) > 0, "Should have CSK matches")
            
            # Check all matches involve CSK
            csk_name = "Chennai Super Kings"
            csk_involved = (
                (csk_data['batting_team'] == csk_name) | 
                (csk_data['bowling_team'] == csk_name)
            ).all()
            
            self.assertTrue(csk_involved, "All matches should involve CSK")
            
        except FileNotFoundError:
            self.skipTest("Raw data file not found")
    
    def test_data_cleaning(self):
        """Test data cleaning process"""
        # Create sample dirty data
        dirty_data = pd.DataFrame({
            'match_id': [1, 2, 3, 3],  # Duplicate
            'date': ['2023-04-01', '2023-04-02', 'invalid_date', '2023-04-04'],
            'batting_team': ['CSK ', ' MI', 'RCB', 'CSK'],  # Extra spaces
            'bowling_team': ['MI', 'CSK', 'MI', 'RCB'],
            'venue': ['Stadium A', None, 'Stadium C', 'Stadium A'],  # Missing value
            'runs': [180, 200, np.nan, 175]  # Missing numerical value
        })
        
        cleaned_data = self.transformer.clean_data(dirty_data)
        
        # Check duplicates removed
        self.assertEqual(len(cleaned_data), 3, "Duplicates should be removed")
        
        # Check strings normalized
        self.assertEqual(cleaned_data['batting_team'].iloc[0], 'CSK', "Strings should be normalized")
        
        # Check missing values handled
        self.assertFalse(cleaned_data['venue'].isnull().any(), "Missing categorical values should be filled")
        self.assertFalse(cleaned_data['runs'].isnull().any(), "Missing numerical values should be filled")
    
    def test_feature_creation(self):
        """Test feature creation"""
        # Create sample data
        sample_data = pd.DataFrame({
            'match_id': [1, 2, 3],
            'date': ['2023-04-01', '2023-04-02', '2023-04-03'],
            'batting_team': ['Chennai Super Kings', 'Mumbai Indians', 'Chennai Super Kings'],
            'bowling_team': ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore'],
            'venue': ['MA Chidambaram Stadium', 'Wankhede Stadium', 'M Chinnaswamy Stadium'],
            'toss_winner': ['Chennai Super Kings', 'Mumbai Indians', 'Chennai Super Kings'],
            'toss_decision': ['bat', 'field', 'bat'],
            'match_won_by': ['Chennai Super Kings', 'Mumbai Indians', 'Chennai Super Kings']
        })
        
        featured_data = self.transformer.create_csk_features(sample_data)
        
        # Check new features exist
        expected_features = [
            'csk_involved', 'opponent', 'csk_won_toss', 
            'csk_won_match', 'is_home_match'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, featured_data.columns, f"Feature '{feature}' should be created")
        
        # Check feature values
        self.assertEqual(featured_data['csk_involved'].sum(), 3, "All matches should involve CSK")
        self.assertEqual(featured_data['csk_won_match'].sum(), 2, "CSK should win 2 matches")
        self.assertEqual(featured_data['is_home_match'].iloc[0], 1, "First match should be home match")
    
    def test_processed_data_quality(self):
        """Test processed data quality"""
        try:
            processed_data = self.loader.load_processed_data("csk_match_level_data.csv")
            
            # Check data exists
            self.assertTrue(len(processed_data) > 0, "Processed data should not be empty")
            
            # Check target column exists
            self.assertIn('csk_won_match', processed_data.columns, "Target column should exist")
            
            # Check target values are binary
            unique_targets = processed_data['csk_won_match'].unique()
            self.assertTrue(set(unique_targets).issubset({0, 1}), "Target should be binary (0, 1)")
            
            # Check for excessive missing values
            missing_percentage = processed_data.isnull().sum() / len(processed_data)
            excessive_missing = missing_percentage > 0.5
            
            self.assertFalse(excessive_missing.any(), 
                           f"No column should have >50% missing values. Found: {excessive_missing[excessive_missing].index.tolist()}")
            
            # Check data balance
            class_distribution = processed_data['csk_won_match'].value_counts()
            minority_ratio = class_distribution.min() / class_distribution.max()
            
            self.assertGreater(minority_ratio, 0.2, 
                             f"Classes should not be extremely imbalanced. Ratio: {minority_ratio:.3f}")
            
        except FileNotFoundError:
            self.skipTest("Processed data file not found")
    
    def test_data_consistency(self):
        """Test data consistency across pipeline stages"""
        try:
            # Load data from different stages
            raw_data = self.extractor.load_ipl_data()
            csk_raw = self.extractor.extract_csk_matches(raw_data)
            processed_data = self.loader.load_processed_data("csk_match_level_data.csv")
            
            # Check match count consistency
            unique_matches_raw = csk_raw['match_id'].nunique()
            unique_matches_processed = len(processed_data)
            
            # Processed should have fewer or equal matches (due to cleaning)
            self.assertLessEqual(unique_matches_processed, unique_matches_raw,
                               "Processed data should not have more matches than raw data")
            
            # Check date ranges
            if 'date' in processed_data.columns:
                raw_dates = pd.to_datetime(csk_raw['date'], errors='coerce')
                processed_dates = pd.to_datetime(processed_data['date'], errors='coerce')
                
                self.assertGreaterEqual(processed_dates.min(), raw_dates.min(),
                                      "Processed data date range should be within raw data range")
                self.assertLessEqual(processed_dates.max(), raw_dates.max(),
                                   "Processed data date range should be within raw data range")
            
        except FileNotFoundError:
            self.skipTest("Required data files not found")

class TestDataValidation(unittest.TestCase):
    """Test data validation rules"""
    
    def test_match_id_uniqueness(self):
        """Test match ID uniqueness in processed data"""
        try:
            loader = DataLoader("data")
            processed_data = loader.load_processed_data("csk_match_level_data.csv")
            
            # Check match_id uniqueness
            if 'match_id' in processed_data.columns:
                duplicate_matches = processed_data['match_id'].duplicated().sum()
                self.assertEqual(duplicate_matches, 0, "Match IDs should be unique in processed data")
            
        except FileNotFoundError:
            self.skipTest("Processed data file not found")
    
    def test_date_validity(self):
        """Test date column validity"""
        try:
            loader = DataLoader("data")
            processed_data = loader.load_processed_data("csk_match_level_data.csv")
            
            if 'date' in processed_data.columns:
                # Check dates are valid
                dates = pd.to_datetime(processed_data['date'], errors='coerce')
                invalid_dates = dates.isnull().sum()
                
                self.assertEqual(invalid_dates, 0, "All dates should be valid")
                
                # Check date range is reasonable (IPL started in 2008)
                min_year = dates.dt.year.min()
                max_year = dates.dt.year.max()
                
                self.assertGreaterEqual(min_year, 2008, "Dates should not be before IPL started")
                self.assertLessEqual(max_year, 2025, "Dates should not be in far future")
            
        except FileNotFoundError:
            self.skipTest("Processed data file not found")
    
    def test_categorical_values(self):
        """Test categorical column values"""
        try:
            loader = DataLoader("data")
            processed_data = loader.load_processed_data("csk_match_level_data.csv")
            
            # Test toss_decision values
            if 'toss_decision' in processed_data.columns:
                valid_decisions = {'bat', 'field', 'Unknown'}
                actual_decisions = set(processed_data['toss_decision'].unique())
                
                self.assertTrue(actual_decisions.issubset(valid_decisions),
                              f"Invalid toss decisions found: {actual_decisions - valid_decisions}")
            
            # Test binary columns
            binary_columns = ['csk_won_toss', 'csk_won_match', 'is_home_match']
            for col in binary_columns:
                if col in processed_data.columns:
                    unique_values = set(processed_data[col].unique())
                    valid_binary = {0, 1, 0.0, 1.0}
                    
                    self.assertTrue(unique_values.issubset(valid_binary),
                                  f"Column '{col}' should only contain binary values. Found: {unique_values}")
            
        except FileNotFoundError:
            self.skipTest("Processed data file not found")

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
