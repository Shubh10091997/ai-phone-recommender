"""
Flask API for Phone Recommendation Engine
Endpoint: http://localhost:5001/api/recommend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import sys

# Add project to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from recommender import PhoneRecommender, train_model

app = Flask(__name__)
CORS(app)  # Enable CORS for website integration

# Load model
MODEL_PATH = BASE_DIR / "models" / "phone_recommender.pkl"
DATA_PATH = BASE_DIR / "data" / "phones_data.csv"

recommender = None

def initialize_model():
    """Load or train the model"""
    global recommender
    
    if MODEL_PATH.exists():
        print("📦 Loading existing model...")
        recommender = PhoneRecommender()
        recommender = recommender.load_model(MODEL_PATH)
    else:
        print("🤖 Training new model...")
        # First prepare data if needed
        if not DATA_PATH.exists():
            print("📊 Preparing data...")
            from prepare_data import save_data_csv
            save_data_csv()
        
        recommender = train_model()

# ===========================
# ROOT PAGE
# ===========================
@app.route("/", methods=["GET"])
def home():
    """Simple front page for quick checks"""
    return (
        "<h2>AI Phone Recommender API</h2>"
        "<p>Use these endpoints:</p>"
        "<ul>"
        "<li><a href='/health'>/health</a></li>"
        "<li><a href='/api/search?q=gaming+under+20000'>/api/search?q=gaming+under+20000</a></li>"
        "<li>POST /api/recommend</li>"
        "<li><a href='/api/stats'>/api/stats</a></li>"
        "</ul>"
    )

# ===========================
# HEALTH CHECK
# ===========================
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "phone-recommender-api"}), 200

# ===========================
# RECOMMENDATION ENDPOINTS
# ===========================

@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    Main recommendation endpoint
    
    Request body:
    {
        "query": "gaming phone under 20000",  # OR
        "use_case": "gaming",
        "budget": 20000,
        "top_k": 5
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        query = data.get("query")
        use_case = data.get("use_case")
        budget = data.get("budget", type=int) if isinstance(data.get("budget"), int) else None
        top_k = data.get("top_k", 5)
        
        # Validate inputs
        if not query and not use_case and not budget:
            return jsonify({
                "error": "Provide at least one of: query, use_case, or budget"
            }), 400
        
        results = []
        
        # 1. Text-based recommendation (if query provided)
        if query:
            results = recommender.recommend_by_text(query, top_k=top_k)
        
        # 2. Specs-based recommendation (if use_case or budget provided)
        elif use_case or budget:
            results = recommender.recommend_by_specs(
                budget=budget,
                use_case=use_case,
                top_k=top_k
            )
        
        return jsonify({
            "success": True,
            "count": len(results),
            "recommendations": results
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Error processing recommendation request"
        }), 500

@app.route("/api/search", methods=["GET"])
def search():
    """
    Search phones by query string
    Usage: /api/search?q=gaming+under+20k
    """
    try:
        q = request.args.get("q", "")
        top_k = request.args.get("top_k", 5, type=int)
        
        if not q:
            return jsonify({"error": "Query parameter 'q' is required"}), 400
        
        results = recommender.recommend_by_text(q, top_k=top_k)
        
        return jsonify({
            "success": True,
            "query": q,
            "count": len(results),
            "recommendations": results
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route("/api/filter", methods=["GET"])
def filter_phones():
    """
    Filter phones by specs
    Usage: /api/filter?budget=20000&use_case=gaming&top_k=5
    """
    try:
        budget = request.args.get("budget", type=int)
        use_case = request.args.get("use_case")
        top_k = request.args.get("top_k", 5, type=int)
        
        if not budget and not use_case:
            return jsonify({
                "error": "Provide at least one of: budget or use_case"
            }), 400
        
        results = recommender.recommend_by_specs(
            budget=budget,
            use_case=use_case,
            top_k=top_k
        )
        
        return jsonify({
            "success": True,
            "filters": {
                "budget": budget,
                "use_case": use_case
            },
            "count": len(results),
            "recommendations": results
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route("/api/stats", methods=["GET"])
def stats():
    """Get dataset statistics"""
    try:
        stats_data = {
            "total_phones": len(recommender.df),
            "brands": recommender.df['brand'].unique().tolist(),
            "price_range": {
                "min": int(recommender.df['price'].min()),
                "max": int(recommender.df['price'].max()),
                "avg": int(recommender.df['price'].mean())
            },
            "use_cases": recommender.df['best_for'].unique().tolist()[:10]
        }
        
        return jsonify({
            "success": True,
            "stats": stats_data
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# ERROR HANDLERS
# ===========================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("🚀 Initializing Phone Recommender API...\n")
    initialize_model()
    print("\n✅ API Ready!")
    print("📍 Running on http://localhost:5001")
    print("\nAvailable endpoints:")
    print("  - POST /api/recommend")
    print("  - GET  /api/search")
    print("  - GET  /api/filter")
    print("  - GET  /api/stats")
    print("  - GET  /health\n")
    
    app.run(host="0.0.0.0", port=5001, debug=True)
