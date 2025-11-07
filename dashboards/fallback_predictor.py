"""
Fallback predictor for when model files are not available
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

class FallbackPredictor:
    """Simple fallback predictor based on historical CSK performance"""
    
    def __init__(self):
        # Historical CSK performance factors
        self.base_win_rate = 0.56  # CSK's historical win rate
        
        # Factor adjustments based on match conditions
        self.factors = {
            'home_advantage': 0.15,      # 15% boost at home
            'toss_advantage': 0.08,      # 8% boost when winning toss
            'peak_season': 0.05,         # 5% boost in peak seasons
            'strong_opponent': -0.10,    # 10% penalty vs strong opponents
            'playoff_pressure': -0.05    # 5% penalty in playoffs
        }
    
    def get_prediction_explanation(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction based on simple rules"""
        
        # Start with base win rate
        win_probability = self.base_win_rate
        key_factors = {}
        
        # Apply adjustments
        
        # Home advantage
        if self._is_home_venue(match_data.get('venue', ''), match_data.get('city', '')):
            win_probability += self.factors['home_advantage']
            key_factors['home_advantage'] = 'Playing at home venue (Chennai/Chepauk)'
        
        # Toss advantage
        if match_data.get('toss_winner') == 'Chennai Super Kings':
            win_probability += self.factors['toss_advantage']
            key_factors['toss_advantage'] = 'Won the toss'
        
        # Peak season
        if self._is_peak_season(match_data.get('season', 2025)):
            win_probability += self.factors['peak_season']
            key_factors['peak_season'] = 'Peak performance season for CSK'
        
        # Strong opponent
        if self._is_strong_opponent(match_data.get('opponent', '')):
            win_probability += self.factors['strong_opponent']
            key_factors['strong_opponent'] = f"Facing strong opponent: {match_data.get('opponent', '')}"
        
        # Playoff pressure
        if self._is_playoff_match(match_data.get('stage', '')):
            win_probability += self.factors['playoff_pressure']
            key_factors['playoff_match'] = 'High-stakes playoff match'
        
        # Ensure probability is within bounds
        win_probability = max(0.1, min(0.9, win_probability))
        
        # Determine prediction
        prediction = 'WIN' if win_probability > 0.5 else 'LOSS'
        confidence = max(win_probability, 1 - win_probability)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'win_probability': win_probability,
            'loss_probability': 1 - win_probability,
            'key_factors': key_factors,
            'model_info': {
                'model_name': 'Fallback Rule-Based Model',
                'training_accuracy': 0.56
            }
        }
    
    def _is_home_venue(self, venue: str, city: str) -> bool:
        """Check if match is at home venue"""
        home_indicators = ['chennai', 'chepauk', 'ma chidambaram']
        venue_lower = venue.lower()
        city_lower = city.lower()
        
        return any(indicator in venue_lower or indicator in city_lower 
                  for indicator in home_indicators)
    
    def _is_peak_season(self, season: int) -> bool:
        """Check if season is a peak performance season for CSK"""
        peak_seasons = [2010, 2011, 2018, 2021, 2023]
        return season in peak_seasons
    
    def _is_strong_opponent(self, opponent: str) -> bool:
        """Check if opponent is considered strong"""
        strong_opponents = [
            'Mumbai Indians', 'Royal Challengers Bangalore',
            'Kolkata Knight Riders', 'Delhi Capitals'
        ]
        return opponent in strong_opponents
    
    def _is_playoff_match(self, stage: str) -> bool:
        """Check if match is a playoff match"""
        playoff_stages = ['qualifier1', 'qualifier2', 'eliminator', 'final']
        return stage.lower() in playoff_stages
