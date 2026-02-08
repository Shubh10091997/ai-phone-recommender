# 🤖 AI Phone Recommender Engine

Intelligent phone recommendation system using TF-IDF + Cosine Similarity

## 📋 Project Structure

```
ai-phone-recommender/
├── app.py              # Flask API
├── recommender.py      # ML recommendation engine
├── prepare_data.py     # Data preparation script
├── requirements.txt    # Dependencies
├── data/
│   └── phones_data.csv # Extracted phone data
└── models/
    └── phone_recommender.pkl # Trained model
```

## 🚀 Setup & Installation

### 1. Install Dependencies
```powershell
cd c:\Users\Shubham\Desktop\ai-phone-recommender
pip install -r requirements.txt
```

### 2. Prepare Data
Extract phone data from your JSON files:
```powershell
python prepare_data.py
```

Output:
- ✅ Creates `data/phones_data.csv` with all phone data
- 📊 Shows total phones and brands

### 3. Train Model
Train the recommendation model:
```powershell
python recommender.py
```

Output:
- 🤖 Creates `models/phone_recommender.pkl`
- ✅ Ready for inference

### 4. Start API Server
```powershell
python app.py
```

Server runs on: **http://localhost:5001**

## 📡 API Endpoints

### 1. Text-Based Search
```
POST /api/recommend
Content-Type: application/json

{
    "query": "gaming phone under 20000",
    "top_k": 5
}
```

### 2. Filter by Specs
```
GET /api/filter?budget=20000&use_case=gaming&top_k=5
```

### 3. Quick Search
```
GET /api/search?q=best+camera+phone
```

### 4. Dataset Stats
```
GET /api/stats
```

### 5. Health Check
```
GET /health
```

## 💻 Example Response

```json
{
    "success": true,
    "count": 5,
    "recommendations": [
        {
            "model": "Poco X6",
            "brand": "Poco",
            "price": 19999,
            "rating": 4.5,
            "processor": "Snapdragon 7 Gen 1",
            "best_for": "gaming, value",
            "similarity_score": 0.85
        },
        ...
    ]
}
```

## 🔗 Website Integration

In your website's `index.html`:

```html
<!-- Quick Phone Search with AI -->
<div id="ai-search">
    <input type="text" id="ai-query" placeholder="e.g., gaming under 20k">
    <button onclick="searchWithAI()">🤖 AI Search</button>
    <div id="ai-results"></div>
</div>

<script>
async function searchWithAI() {
    const query = document.getElementById('ai-query').value;
    const response = await fetch('http://localhost:5001/api/recommend', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: query, top_k: 5})
    });
    const data = await response.json();
    // Display results...
}
</script>
```

## 🎯 How It Works

1. **Data Extraction**: Extract features from your phone JSON files
2. **Vectorization**: Convert text to TF-IDF vectors
3. **Similarity**: Calculate cosine similarity between queries and phones
4. **Ranking**: Return top-K matching phones
5. **API**: Expose recommendations via REST API

## 📊 Features Used

- Brand
- Processor
- Best For (use case)
- Price
- RAM/Storage
- Camera, Gaming, Battery, Performance ratings
- Launch Year

## ⚙️ Configuration

To change settings, edit:

**app.py**
- Port: Change `port=5001`
- Debug: Set `debug=False` for production

**recommender.py**
- Top K: Default is 5, customize in endpoint calls
- Vectorizer: Adjust `max_features=100`

## 🐛 Troubleshooting

**Issue**: "Data file not found"
```
Solution: Run python prepare_data.py first
```

**Issue**: "Model not trained"
```
Solution: Run python recommender.py first
```

**Issue**: CORS errors on website
```
Solution: app.py already has CORS enabled (flask-cors)
```

## 📝 Notes

- First run: Data preparation + Model training takes ~1-2 minutes
- Subsequent runs: Model loads from pickle file (instant)
- No external API keys required
- All processing done locally on your machine

## ✨ Future Enhancements

- [ ] Add collaborative filtering
- [ ] Implement Matrix Factorization
- [ ] Add user preference learning
- [ ] Create web UI dashboard
- [ ] Add Docker containerization
