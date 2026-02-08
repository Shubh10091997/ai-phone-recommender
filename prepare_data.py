"""
Data Preparation Script
Extract phone data from JSON files and prepare for AI model
"""

import json
import os
import pandas as pd
from pathlib import Path

# Get paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TOP3_DATA_DIR = Path(r"c:\Users\Shubham\Desktop\top3-mobile\data")

def extract_phone_data():
    """Extract all phone data from JSON files"""
    brands = ["samsung", "realme", "redmi", "poco", "apple", "vivo", "oppo", "motorola"]
    all_phones = []
    
    for brand in brands:
        brand_file = TOP3_DATA_DIR / f"{brand}.json"
        try:
            with open(brand_file, "r", encoding="utf-8") as f:
                brand_data = json.load(f)
                
            if isinstance(brand_data, dict) and "phones" in brand_data:
                phones = brand_data.get("phones", [])
                
                for phone in phones:
                    # Extract key features
                    phone_record = {
                        "id": phone.get("id", ""),
                        "model": phone.get("model", ""),
                        "brand": brand_data.get("brand", brand.capitalize()),
                        "price": phone.get("price", 0),
                        "launch_year": phone.get("launch_year", 0),
                        "processor": phone.get("processor", ""),
                        "ram": phone.get("ram", [""])[0] if isinstance(phone.get("ram"), list) else phone.get("ram", ""),
                        "storage": phone.get("storage", [""])[0] if isinstance(phone.get("storage"), list) else phone.get("storage", ""),
                        "display": str(phone.get("display", {}).get("size", "")) if isinstance(phone.get("display"), dict) else "",
                        "battery": str(phone.get("battery", {}).get("capacity", "")) if isinstance(phone.get("battery"), dict) else "",
                        "best_for": ", ".join(phone.get("best_for", [])) if isinstance(phone.get("best_for"), list) else "",
                        "gaming": phone.get("gaming", 4.0),
                        "camera": phone.get("camera", 4.0),
                        "battery_score": phone.get("battery_score", 4.0) if "battery_score" in phone else 4.0,
                        "performance": phone.get("performance", 4.0),
                        "display_score": phone.get("display", 4.0) if isinstance(phone.get("display"), (int, float)) else 4.0,
                        "reason": phone.get("reason", ""),
                        "rating": phone.get("rating", 4.0),
                    }
                    all_phones.append(phone_record)
                    
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading {brand}.json: {e}")
    
    return all_phones

def save_data_csv():
    """Save extracted data as CSV"""
    phones = extract_phone_data()
    
    if not phones:
        print("❌ No phone data extracted!")
        return False
    
    df = pd.DataFrame(phones)
    
    # Save as CSV
    csv_path = DATA_DIR / "phones_data.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    
    print(f"✅ Data saved: {csv_path}")
    print(f"📊 Total phones: {len(df)}")
    print(f"📱 Brands: {', '.join(df['brand'].unique())}")
    
    return True

if __name__ == "__main__":
    save_data_csv()
