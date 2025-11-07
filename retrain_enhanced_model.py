"""
Enhanced Model Training Script
Retrain CSK prediction model with improved accuracy targeting 75%+
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb

class EnhancedCSKTrainer:
    """Enhanced training pipeline for better accuracy"""
    
    def __init__(self):
        self.models_dir = Path("models/artifacts")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.venue_encoder = LabelEncoder()
        self.opponent_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        self.best_model = None
        self.best_accuracy = 0
        
    def load_and_prepare_data(self):
        """Load and prepare training data with enhanced features"""
        
        # Load processed CSK match data
        try:
            df = pd.read_csv("data/processed/csk_match_level_data.csv")
            print(f"✅ Loaded {len(df)} CSK matches from processed data")
            csk_matches = df.copy()
        except:
            print("❌ Could not load processed CSK data - creating synthetic training data")
            df = self._create_synthetic_data()
            # Filter CSK matches
            csk_matches = df[
                (df['team1'] == 'Chennai Super Kings') | 
                (df['team2'] == 'Chennai Super Kings')
            ].copy()
            print(f"📊 Found {len(csk_matches)} CSK matches")
        
        # Create enhanced features
        features_df = self._create_enhanced_features(csk_matches)
        
        return features_df
    
    def _create_synthetic_data(self):
        """Create synthetic training data if real data not available"""
        np.random.seed(42)
        
        teams = [
            'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
            'Kolkata Knight Riders', 'Delhi Capitals', 'Rajasthan Royals',
            'Punjab Kings', 'Sunrisers Hyderabad', 'Gujarat Titans', 'Lucknow Super Giants'
        ]
        
        venues = [
            'MA Chidambaram Stadium, Chepauk', 'Wankhede Stadium', 'Eden Gardens',
            'M Chinnaswamy Stadium', 'Rajiv Gandhi International Stadium',
            'Sawai Mansingh Stadium', 'Feroz Shah Kotla'
        ]
        
        cities = ['Chennai', 'Mumbai', 'Kolkata', 'Bangalore', 'Hyderabad', 'Jaipur', 'Delhi']
        
        # Generate 500 synthetic matches
        data = []
        for i in range(500):
            team1 = 'Chennai Super Kings'
            team2 = np.random.choice([t for t in teams if t != team1])
            
            match_data = {
                'team1': team1,
                'team2': team2,
                'venue': np.random.choice(venues),
                'city': np.random.choice(cities),
                'season': np.random.choice(range(2008, 2024)),
                'toss_winner': np.random.choice([team1, team2]),
                'toss_decision': np.random.choice(['bat', 'field']),
                'winner': team1 if np.random.random() > 0.44 else team2  # CSK 56% win rate
            }
            data.append(match_data)
        
        return pd.DataFrame(data)
    
    def _create_enhanced_features(self, df):
        """Create enhanced features for better prediction"""
        
        features = []
        
        # Calculate historical performance for each venue
        venue_stats = self._calculate_venue_stats_processed(df)
        
        for _, match in df.iterrows():
            # Skip matches with missing critical data
            if pd.isna(match.get('csk_won_match')):
                continue
            
            # Extract season year from season string (e.g., "2007/08" -> 2008)
            season_str = str(match.get('season', '2020'))
            if '/' in season_str:
                season = int(season_str.split('/')[1]) + 2000 if int(season_str.split('/')[1]) < 50 else int(season_str.split('/')[1]) + 1900
            else:
                season = int(season_str)
            
            # For processed data, we need to infer opponent from the data structure
            # Since this is CSK-specific data, we'll use venue and other info to create features
            
            # Basic features from processed data
            feature_row = {
                'venue': str(match.get('venue', 'Unknown')),
                'city': str(match.get('city', 'Unknown')),
                'season': season,
                'toss_won': int(match.get('csk_won_toss', 0)),
                'chose_to_bat': int(match.get('chose_to_bat', 0)),
                'home_match': int(match.get('is_home_match', 0)),
                'target': int(match.get('csk_won_match', 0))
            }
            
            # Add venue performance
            feature_row['venue_win_rate'] = venue_stats.get(str(match.get('venue', 'Unknown')), 0.5)
            
            # Enhanced features based on available data
            feature_row.update(self._add_enhanced_features_processed(match, season))
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def _calculate_opponent_stats(self, df):
        """Calculate CSK's win rate against each opponent"""
        opponent_stats = {}
        
        for opponent in df['team1'].unique():
            if opponent == 'Chennai Super Kings' or pd.isna(opponent):
                continue
                
            # Get matches against this opponent
            matches = df[
                ((df['team1'] == 'Chennai Super Kings') & (df['team2'] == opponent)) |
                ((df['team2'] == 'Chennai Super Kings') & (df['team1'] == opponent))
            ]
            
            if len(matches) > 0:
                wins = len(matches[matches['winner'] == 'Chennai Super Kings'])
                win_rate = wins / len(matches)
                opponent_stats[opponent] = win_rate
        
        return opponent_stats
    
    def _calculate_venue_stats(self, df):
        """Calculate CSK's win rate at each venue"""
        venue_stats = {}
        
        csk_matches = df[
            (df['team1'] == 'Chennai Super Kings') | 
            (df['team2'] == 'Chennai Super Kings')
        ]
        
        for venue in csk_matches['venue'].unique():
            if pd.isna(venue):
                continue
                
            venue_matches = csk_matches[csk_matches['venue'] == venue]
            if len(venue_matches) > 0:
                wins = len(venue_matches[venue_matches['winner'] == 'Chennai Super Kings'])
                win_rate = wins / len(venue_matches)
                venue_stats[str(venue)] = win_rate
        
        return venue_stats
    
    def _calculate_venue_stats_processed(self, df):
        """Calculate CSK's win rate at each venue from processed data"""
        venue_stats = {}
        
        for venue in df['venue'].unique():
            if pd.isna(venue):
                continue
                
            venue_matches = df[df['venue'] == venue]
            if len(venue_matches) > 0:
                wins = len(venue_matches[venue_matches['csk_won_match'] == 1])
                win_rate = wins / len(venue_matches)
                venue_stats[str(venue)] = win_rate
        
        return venue_stats
    
    def _add_enhanced_features_processed(self, match, season):
        """Add enhanced features for processed data"""
        
        venue = str(match.get('venue', ''))
        city = str(match.get('city', ''))
        
        # Venue performance mapping
        venue_performance = {
            'MA Chidambaram Stadium, Chepauk': 0.78,
            'Wankhede Stadium': 0.42,
            'Eden Gardens': 0.55,
            'M Chinnaswamy Stadium': 0.46,
            'Rajiv Gandhi International Stadium': 0.62,
            'Sawai Mansingh Stadium': 0.65,
            'Feroz Shah Kotla': 0.57
        }
        
        enhanced_features = {
            'venue_performance': venue_performance.get(venue, 0.56),
            'peak_season': 1 if season in [2010, 2011, 2018, 2021, 2023] else 0,
            'recent_season': 1 if season >= 2018 else 0,
            'dhoni_era': 1 if season <= 2023 else 0,
            'season_normalized': (season - 2008) / (2023 - 2008) if season >= 2008 else 0,
            'chepauk_match': 1 if 'chepauk' in venue.lower() else 0
        }
        
        return enhanced_features
    
    def _add_enhanced_features(self, match, opponent):
        """Add enhanced features for better accuracy"""
        
        # Opponent strength mapping
        opponent_strength = {
            'Mumbai Indians': 0.85, 'Royal Challengers Bangalore': 0.75,
            'Kolkata Knight Riders': 0.70, 'Delhi Capitals': 0.68,
            'Sunrisers Hyderabad': 0.65, 'Rajasthan Royals': 0.60,
            'Punjab Kings': 0.55, 'Gujarat Titans': 0.72,
            'Lucknow Super Giants': 0.68
        }
        
        # Venue performance for CSK
        venue_performance = {
            'MA Chidambaram Stadium, Chepauk': 0.78,
            'Wankhede Stadium': 0.42,
            'Eden Gardens': 0.55,
            'M Chinnaswamy Stadium': 0.46,
            'Rajiv Gandhi International Stadium': 0.62,
            'Sawai Mansingh Stadium': 0.65,
            'Feroz Shah Kotla': 0.57
        }
        
        season = match.get('season', 2020)
        venue = match.get('venue', '')
        
        enhanced_features = {
            'opponent_strength': opponent_strength.get(opponent, 0.60),
            'venue_performance': venue_performance.get(venue, 0.56),
            'peak_season': 1 if season in [2010, 2011, 2018, 2021, 2023] else 0,
            'recent_season': 1 if season >= 2018 else 0,
            'dhoni_era': 1 if season <= 2023 else 0,
            'season_normalized': (season - 2008) / (2023 - 2008),
            'rivalry_match': 1 if opponent in ['Mumbai Indians', 'Royal Challengers Bangalore'] else 0
        }
        
        return enhanced_features
    
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            print(f"   Accuracy: {accuracy:.3f}")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall: {recall:.3f}")
            print(f"   F1-Score: {f1:.3f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_name = name
        
        print(f"\n🏆 Best Model: {best_name} with {best_score:.3f} accuracy")
        
        # Create ensemble if accuracy is still low
        if best_score < 0.70:
            print("\n🔄 Creating ensemble model for better accuracy...")
            ensemble = self._create_ensemble(models, X_train, y_train, X_test, y_test)
            if ensemble['accuracy'] > best_score:
                best_model = ensemble['model']
                best_score = ensemble['accuracy']
                best_name = "Ensemble"
                print(f"🎯 Ensemble accuracy: {best_score:.3f}")
        
        self.best_model = best_model
        self.best_accuracy = best_score
        
        return best_model, best_score, results
    
    def _create_ensemble(self, models, X_train, y_train, X_test, y_test):
        """Create ensemble model for better accuracy"""
        
        # Select top 3 models for ensemble
        ensemble_models = [
            ('rf', models['Random Forest']),
            ('xgb', models['XGBoost']),
            ('lgb', models['LightGBM'])
        ]
        
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'model': ensemble,
            'accuracy': accuracy
        }
    
    def save_model(self, model, encoders, feature_names):
        """Save the trained model and encoders"""
        
        # Save model
        model_path = self.models_dir / "csk_best_model_random_forest.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Save encoders
        venue_encoder_path = self.models_dir / "venue_encoder.pkl"
        joblib.dump(encoders['venue'], venue_encoder_path)
        
        opponent_encoder_path = self.models_dir / "opponent_encoder.pkl"
        joblib.dump(encoders['opponent'], opponent_encoder_path)
        
        print(f"✅ Encoders saved")
        
        # Save metadata
        metadata = {
            'model_type': type(model).__name__,
            'accuracy': self.best_accuracy,
            'feature_names': feature_names,
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = self.models_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Metadata saved to {metadata_path}")

def main():
    """Main training pipeline"""
    print("🚀 Starting Enhanced CSK Model Training...")
    
    trainer = EnhancedCSKTrainer()
    
    # Load and prepare data
    print("\n📊 Loading and preparing data...")
    df = trainer.load_and_prepare_data()
    
    # Prepare features
    print("\n🔧 Preparing features...")
    
    # Encode categorical variables
    df['venue_encoded'] = trainer.venue_encoder.fit_transform(df['venue'])
    
    # Select features (adapted for processed data)
    feature_columns = [
        'venue_encoded', 'season', 'toss_won', 'chose_to_bat',
        'home_match', 'venue_performance', 'peak_season',
        'recent_season', 'dhoni_era', 'season_normalized', 'chepauk_match',
        'venue_win_rate'  # Historical venue performance
    ]
    
    X = df[feature_columns]
    y = df['target']
    
    print(f"📊 Features: {len(feature_columns)}")
    print(f"📊 Samples: {len(X)}")
    print(f"📊 Win rate: {y.mean():.3f}")
    
    # Train models
    print("\n🎯 Training models...")
    best_model, best_accuracy, results = trainer.train_models(X, y)
    
    # Save model if accuracy is acceptable
    if best_accuracy >= 0.60:
        print(f"\n💾 Saving model with {best_accuracy:.3f} accuracy...")
        
        encoders = {
            'venue': trainer.venue_encoder,
            'opponent': trainer.opponent_encoder
        }
        
        trainer.save_model(best_model, encoders, feature_columns)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"🎯 Final accuracy: {best_accuracy:.3f}")
        
        if best_accuracy >= 0.75:
            print("🏆 Target accuracy of 75% achieved!")
        elif best_accuracy >= 0.70:
            print("✅ Good accuracy achieved!")
        else:
            print("⚠️ Consider collecting more data or feature engineering for better accuracy")
    
    else:
        print(f"\n❌ Model accuracy too low: {best_accuracy:.3f}")
        print("💡 Consider collecting more training data or improving features")

if __name__ == "__main__":
    main()
