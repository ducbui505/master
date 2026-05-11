"""
fetch_pubchem.py - Lay mo ta thuoc tu PubChem API (mien phi, khong can dang ky)

Input : data/drug_mapping.csv  (idx, drugbank_id, drug_name)
Output: data/drugbank_text.csv (drugbank_id, drug_name, description, moa)

PubChem PUG REST API:
    https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/description/JSON

Chay:
    python prepare_data/fetch_pubchem.py
"""

import os
import time
import json
import pandas as pd
import urllib.request
import urllib.parse
import urllib.error

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, "data")
OUTPUT = os.path.join(DATA, "drugbank_text.csv")

CACHE_FILE = os.path.join(DATA, "pubchem_cache.json")  # cache de tranh query lai

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def fetch_description(drug_name, cache):
    """Query PubChem description cho 1 ten thuoc. Tra ve (description, moa)."""
    if drug_name in cache:
        return cache[drug_name]

    encoded = urllib.parse.quote(drug_name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/description/JSON"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        descriptions = data.get("InformationList", {}).get("Information", [])
        desc_text = ""
        for info in descriptions:
            if "Description" in info:
                candidate = str(info["Description"]).strip()
                # Lay doan mo ta dai nhat
                if len(candidate) > len(desc_text):
                    desc_text = candidate

        result = (desc_text, "")
        cache[drug_name] = result
        return result

    except urllib.error.HTTPError:
        result = ("", "")
        cache[drug_name] = result
        return result
    except Exception:
        return ("", "")

def main():
    os.makedirs(DATA, exist_ok=True)

    drug_map = pd.read_csv(os.path.join(DATA, "drug_mapping.csv"))
    print(f"Tong so thuoc: {len(drug_map)}")

    cache = load_cache()
    already = sum(1 for name in drug_map["drug_name"] if name in cache)
    print(f"Da co trong cache: {already}/{len(drug_map)}")

    records = []
    for i, row in drug_map.iterrows():
        drug_name = str(row["drug_name"]).strip()
        drugbank_id = str(row["drugbank_id"]).strip()

        from_cache = drug_name in cache
        desc, moa = fetch_description(drug_name, cache)

        records.append({
            "drugbank_id" : drugbank_id,
            "drug_name"   : drug_name,
            "description" : desc,
            "moa"         : moa,
        })

        # In tien do moi 10 thuoc
        if (i + 1) % 10 == 0:
            has_desc = sum(1 for r in records if r["description"])
            print(f"  [{i+1}/{len(drug_map)}] co description: {has_desc}")
            save_cache(cache)  # luu cache dinh ky

        # Rate limit: chi sleep khi thuc su goi API (khong phai cache)
        if not from_cache:
            time.sleep(0.22)

    # Luu cache lan cuoi
    save_cache(cache)

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT, index=False, encoding="utf-8")

    has_desc = df["description"].ne("").sum()
    print(f"\nKet qua:")
    print(f"  Tong thuoc      : {len(df)}")
    print(f"  Co description  : {has_desc} ({has_desc/len(df)*100:.1f}%)")
    print(f"  Khong co        : {len(df) - has_desc}")
    print(f"Saved -> {OUTPUT}")
    print("\nSample:")
    print(df[df["description"].ne("")].head(3)[["drug_name","description"]].to_string())

if __name__ == "__main__":
    main()
