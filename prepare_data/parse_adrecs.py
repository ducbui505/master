"""
parse_adrecs.py - Parse ADReCS ontology xlsx -> adrecs_text.csv

Input : ADR_ontology_v3.3.xlsx  (o thu muc goc project)
        Columns: ADRECS_ID, ADR_ID, ADR_TERM, ADR_SYNONYMS, MEDDRA_CODE
Output: data/adrecs_text.csv   (adr_term, meddra_code, adr_synonyms)

Chay:
    python prepare_data/parse_adrecs.py
"""

import os
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH  = os.path.join(BASE, "ADR_ontology_v3.3.xlsx")
OUTPUT   = os.path.join(BASE, "data", "adrecs_text.csv")

NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

def read_xlsx_via_xml(path):
    """Doc xlsx bang raw XML (tranh loi openpyxl voi file co drawing)."""
    with zipfile.ZipFile(path) as z:
        with z.open("xl/sharedStrings.xml") as f:
            ss_tree = ET.parse(f)
        shared = []
        for si in ss_tree.getroot().findall("x:si", NS):
            t = si.find("x:t", NS)
            if t is None:
                parts = [r.find("x:t", NS) for r in si.findall("x:r", NS)]
                text = "".join(p.text or "" for p in parts if p is not None)
            else:
                text = t.text or ""
            shared.append(text)

        with z.open("xl/worksheets/sheet1.xml") as f:
            ws_tree = ET.parse(f)

    rows_data = []
    for row in ws_tree.getroot().findall(".//x:row", NS):
        row_vals = []
        for c in row.findall("x:c", NS):
            t = c.get("t", "")
            v_elem = c.find("x:v", NS)
            if v_elem is None:
                row_vals.append("")
            elif t == "s":
                row_vals.append(shared[int(v_elem.text)])
            else:
                row_vals.append(v_elem.text or "")
        rows_data.append(row_vals)

    max_cols = max(len(r) for r in rows_data)
    for r in rows_data:
        while len(r) < max_cols:
            r.append("")

    header = rows_data[0]
    df = pd.DataFrame(rows_data[1:], columns=header)
    return df

def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    if not os.path.exists(IN_PATH):
        print(f"[ERROR] Khong tim thay: {IN_PATH}")
        return

    print(f"Doc {IN_PATH} ...")
    df = read_xlsx_via_xml(IN_PATH)

    print(f"Shape  : {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head(5).to_string())

    result = df[["ADR_TERM", "ADR_SYNONYMS", "MEDDRA_CODE"]].copy()
    result.columns = ["adr_term", "adr_synonyms", "meddra_code"]

    result["adr_term_lower"] = result["adr_term"].str.lower().str.strip()
    result["meddra_code"]    = result["meddra_code"].astype(str).str.strip()

    result = result[result["adr_term"].ne("") & result["adr_term"].notna()]

    result.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"\nSaved {len(result)} ADR terms -> {OUTPUT}")
    print(f"  Co synonyms    : {result['adr_synonyms'].ne('Not Available').sum()}")
    print(f"  Co meddra_code : {result['meddra_code'].ne('').sum()}")

if __name__ == "__main__":
    main()
