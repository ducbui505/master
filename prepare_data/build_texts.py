"""
build_texts.py - Ghep mapping + PubChem/ADReCS text -> text CSVs.

Input:
    data/drug_mapping.csv    (idx, drugbank_id, drug_name)
    data/se_mapping.csv      (idx, umls_cui, se_name)
    data/drugbank_text.csv   (drugbank_id, drug_name, description, moa)
    data/adrecs_text.csv     (adr_term, adr_synonyms, meddra_code, adr_term_lower)

Default output:
    data/drug_texts.csv  (idx, drug_name, drug_text)
    data/se_texts.csv    (idx, se_name, se_text)

Chay:
    python prepare_data/build_texts.py
    python prepare_data/build_texts.py --drug-mapping data/benchmark_drug_mapping.csv \
        --se-mapping data/benchmark_se_mapping.csv \
        --drug-output data/benchmark_drug_texts.csv \
        --se-output data/benchmark_se_texts.csv
"""

import argparse
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

def parse_args():
    parser = argparse.ArgumentParser(description="Build drug/SE texts from mapping files.")
    parser.add_argument("--drug-mapping", default=os.path.join(DATA, "drug_mapping.csv"),
                        help="Path to drug mapping CSV.")
    parser.add_argument("--se-mapping", default=os.path.join(DATA, "se_mapping.csv"),
                        help="Path to side-effect mapping CSV.")
    parser.add_argument("--drug-output", default=os.path.join(DATA, "drug_texts.csv"),
                        help="Output path for drug texts CSV.")
    parser.add_argument("--se-output", default=os.path.join(DATA, "se_texts.csv"),
                        help="Output path for side-effect texts CSV.")
    return parser.parse_args()


def main():
    args = parse_args()

    drug_map = pd.read_csv(args.drug_mapping)
    se_map   = pd.read_csv(args.se_mapping)
    print(f"drug_mapping: {len(drug_map)} drugs -> {args.drug_mapping}")
    print(f"se_mapping  : {len(se_map)} SEs -> {args.se_mapping}")

    # -- Drug texts (PubChem) --------------------------------------------------
    drugbank_path = os.path.join(DATA, "drugbank_text.csv")
    if os.path.exists(drugbank_path):
        drugbank = pd.read_csv(drugbank_path)
        drugs = drug_map.merge(drugbank[["drugbank_id", "description", "moa"]],
                               on="drugbank_id", how="left")
        print(f"\nPubChem matched: {drugs['description'].notna().sum()}/{len(drugs)} co description")

        def make_drug_text(row):
            parts = []
            if pd.notna(row.get("description")) and str(row["description"]).strip():
                parts.append(str(row["description"]).strip()[:800])
            if pd.notna(row.get("moa")) and str(row["moa"]).strip():
                parts.append(str(row["moa"]).strip()[:400])
            return " ".join(parts) if parts else str(row["drug_name"])

        drugs["drug_text"] = drugs.apply(make_drug_text, axis=1)
    else:
        print("\n[WARNING] drugbank_text.csv khong ton tai -> dung ten thuoc")
        drugs = drug_map.copy()
        drugs["drug_text"] = drugs["drug_name"]

    drug_out = drugs[["idx", "drug_name", "drug_text"]].copy()
    drug_out.to_csv(args.drug_output, index=False, encoding="utf-8")

    full = drug_out["drug_text"].str.len() > drug_out["drug_name"].str.len()
    print(f"Drug texts co mo ta day du : {full.sum()}/{len(drug_out)}")
    print(f"Drug texts fallback (ten)  : {(~full).sum()}/{len(drug_out)}")
    print(f"Saved -> {args.drug_output}")

    # -- SE texts (ADReCS) -----------------------------------------------------
    adrecs_path = os.path.join(DATA, "adrecs_text.csv")
    ses = se_map.copy()

    if os.path.exists(adrecs_path):
        adrecs = pd.read_csv(adrecs_path, dtype=str).fillna("")

        # Chuan hoa meddra_code (bo so 0 dau, vi umls_cui la C + 7 chu so)
        # ADReCS MEDDRA_CODE la so nguyen (vd: 10021085)
        # se_mapping umls_cui la C0021085 (co chu C va 7 chu so, co the co leading zeros)
        # => lay phan so tu umls_cui de so sanh
        ses["umls_num"] = ses["umls_cui"].str.replace("C", "", regex=False).str.lstrip("0")
        adrecs["meddra_num"] = adrecs["meddra_code"].str.lstrip("0")

        # Match theo meddra_code
        merged = ses.merge(
            adrecs[["adr_term", "adr_synonyms", "meddra_num"]].drop_duplicates("meddra_num"),
            left_on="umls_num", right_on="meddra_num", how="left"
        )
        matched = merged["adr_term"].notna().sum()
        print(f"\nADReCS match theo meddra_code: {matched}/{len(ses)}")

        # Fallback: match theo ten (case-insensitive) voi nhung cai chua match
        if matched < len(ses):
            adrecs["adr_term_lower"] = adrecs["adr_term"].str.lower().str.strip()
            merged["se_name_lower"] = merged["se_name"].str.lower().str.strip()
            
            unmatched_mask = merged["adr_term"].isna()
            unmatched = merged[unmatched_mask][["idx", "se_name", "se_name_lower"]].copy()
            fallback = unmatched.merge(
                adrecs[["adr_term", "adr_synonyms", "adr_term_lower"]].drop_duplicates("adr_term_lower"),
                left_on="se_name_lower", right_on="adr_term_lower", how="left"
            )
            # Cap nhat vao merged
            for col in ["adr_term", "adr_synonyms"]:
                merged.loc[unmatched_mask, col] = fallback[col].values
            
            extra = merged["adr_term"].notna().sum() - matched
            print(f"  + fallback match theo ten: +{extra} -> tong {merged['adr_term'].notna().sum()}/{len(ses)}")

        def make_se_text(row):
            name = str(row["se_name"]).strip()
            term = row.get("adr_term", "")
            syns = row.get("adr_synonyms", "")
            if pd.notna(term) and str(term).strip():
                text = str(term).strip()
                if pd.notna(syns) and str(syns).strip() and str(syns) != "Not Available":
                    # Lay toi da 3 synonym dau
                    syn_list = [s.strip() for s in str(syns).split("|")][:3]
                    text += ". Also known as: " + ", ".join(syn_list)
                return text
            return name  # fallback: ten SE

        merged["se_text"] = merged.apply(make_se_text, axis=1)
        se_out = merged[["idx", "se_name", "se_text"]].copy()
    else:
        print("\n[WARNING] adrecs_text.csv khong ton tai -> dung ten SE")
        ses["se_text"] = ses["se_name"]
        se_out = ses[["idx", "se_name", "se_text"]].copy()

    se_out.to_csv(args.se_output, index=False, encoding="utf-8")

    full_se = se_out["se_text"].str.len() > se_out["se_name"].str.len() + 2
    print(f"SE texts co ADReCS term    : {full_se.sum()}/{len(se_out)}")
    print(f"SE texts fallback (ten)    : {(~full_se).sum()}/{len(se_out)}")
    print(f"Saved -> {args.se_output}")

    print("\nSample drug_texts:")
    print(drug_out.head(3)[["drug_name","drug_text"]].to_string())
    print("\nSample se_texts:")
    print(se_out.head(3)[["se_name","se_text"]].to_string())

if __name__ == "__main__":
    main()
