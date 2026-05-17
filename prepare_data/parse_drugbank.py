"""
parse_drugbank.py – Parse DrugBank full XML → drugbank_text.csv

Input : Datas/drugbank_all_full_database.xml  (tải từ drugbank.com)
Output: data/drugbank_text.csv  (drugbank_id, drug_name, description, moa)

Chạy:
    python prepare_data/parse_drugbank.py
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(BASE, "full database.xml")
OUTPUT   = os.path.join(BASE, "data", "drugbank_text.csv")

NS = {'db': 'http://www.drugbank.ca'}

def find_text(elem, tag):
    """Lấy text của 1 tag con, trả về '' nếu không tìm thấy."""
    child = elem.find(tag, NS)
    if child is not None and child.text:
        return child.text.strip()
    return ''

def main():
    if not os.path.exists(XML_PATH):
        print(f"[ERROR] Không tìm thấy: {XML_PATH}")
        print("Hãy tải drugbank_all_full_database.xml từ https://go.drugbank.com/releases")
        print("và đặt vào thư mục Datas/")
        return

    print(f"Parsing {XML_PATH} ...")
    print("(Có thể mất 1-3 phút do file ~1.5GB)")

    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    records = []
    for drug in root.findall('db:drug', NS):
        # Lấy DrugBank ID primary
        dbid = ''
        for id_elem in drug.findall('db:drugbank-id', NS):
            if id_elem.get('primary') == 'true':
                dbid = id_elem.text.strip() if id_elem.text else ''
                break

        name        = find_text(drug, 'db:name')
        description = find_text(drug, 'db:description')
        moa         = find_text(drug, 'db:mechanism-of-action')

        records.append({
            'drugbank_id': dbid,
            'drug_name'  : name,
            'description': description,
            'moa'        : moa,
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding='utf-8')

    print(f"\nParsed {len(df)} drugs")
    print(f"  có description : {df['description'].ne('').sum()}")
    print(f"  có moa         : {df['moa'].ne('').sum()}")
    print(f"Saved → {OUTPUT}")
    print("\nSample:")
    print(df[df['description'].ne('')].head(3)[['drugbank_id','drug_name','description']].to_string())

if __name__ == '__main__':
    main()
