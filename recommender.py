"""
AI Phone Recommendation Engine
Uses TF-IDF + Cosine Similarity for intelligent recommendations
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

class PhoneRecommender:
    def __init__(self):
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.feature_matrix = None
        
    def load_data(self, csv_path):
        """Load phone data from CSV"""
        self.df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(self.df)} phones")
        return self.df
    
    def prepare_features(self):
        """Prepare feature matrix for similarity matching"""
        # Create text features from best_for, processor, etc.
        self.df['combo_features'] = (
            self.df['brand'].fillna('') + ' ' +
            self.df['best_for'].fillna('') + ' ' +
            self.df['processor'].fillna('') + ' ' +
            self.df['reason'].fillna('')
        )
        
        # TF-IDF Vectorization
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, lowercase=True)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combo_features'])
        
        # Normalize numeric features
        numeric_features = ['price', 'gaming', 'camera', 'battery_score', 'performance', 'rating']
        feature_df = self.df[numeric_features].copy()
        
        # Convert all columns to numeric and fill missing with default
        for col in feature_df.columns:
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').fillna(4.0)
        
        # Normalize each feature
        for col in feature_df.columns:
            min_val = feature_df[col].min()
            max_val = feature_df[col].max()
            if max_val - min_val > 0:
                feature_df[col] = (feature_df[col] - min_val) / (max_val - min_val)
        
        self.feature_matrix = feature_df.values
        
        print("✅ Features prepared")
    def recommend_by_text(self, query, top_k=5):
        """Recommend phones based on text query"""
        if self.tfidf_matrix is None:
            raise ValueError("Model not trained! Call prepare_features() first")
        
        # Vectorize query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return results
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include if similarity > 0
                results.append({
                    'model': self.df.loc[idx, 'model'],
                    'brand': self.df.loc[idx, 'brand'],
                    'price': int(self.df.loc[idx, 'price']),
                    'rating': float(self.df.loc[idx, 'rating']),
                    'processor': self.df.loc[idx, 'processor'],
                    'best_for': self.df.loc[idx, 'best_for'],
                    'similarity_score': float(similarities[idx])
                })
        
        return results
    
    def recommend_by_specs(self, budget=None, use_case=None, top_k=5):
        """Recommend phones by budget and use case"""
        results_df = self.df.copy()
        
        # Filter by budget
        if budget:
            results_df = results_df[results_df['price'] <= budget]
        
        # Filter by use case
        if use_case and use_case.lower() != 'overall':
            results_df = results_df[results_df['best_for'].str.lower().str.contains(use_case, na=False)]
        
        if len(results_df) == 0:
            return []
        
        # Sort by rating
        results_df = results_df.sort_values('rating', ascending=False).head(top_k)
        
        results = []
        for _, row in results_df.iterrows():
            results.append({
                'model': row['model'],
                'brand': row['brand'],
                'price': int(row['price']),
                'rating': float(row['rating']),
                'processor': row['processor'],
                'best_for': row['best_for'],
                'reason': row['reason']
            })
        
        return results
    
    def save_model(self, model_path=None):
        """Save trained model"""
        if model_path is None:
            model_path = MODELS_DIR / "phone_recommender.pkl"
        
        joblib.dump(self, model_path)
        print(f"✅ Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load trained model"""
        return joblib.load(model_path)

def train_model():
    """Train the recommendation model"""
    print("🤖 Training Phone Recommendation Model...\n")
    
    # Initialize recommender
    recommender = PhoneRecommender()
    
    # Load data
    csv_path = DATA_DIR / "phones_data.csv"
    if not csv_path.exists():
        print(f"❌ Data file not found: {csv_path}")
        print("Run prepare_data.py first!")
        return None
    
    recommender.load_data(csv_path)
    
    # Prepare features
    recommender.prepare_features()
    
    # Save model
    recommender.save_model()
    
    print("\n✅ Model training complete!")
    return recommender

if __name__ == "__main__":
    train_model()
