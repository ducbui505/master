"""
create_masks.py – Tạo binary mask cho LLM features từ drug_texts.csv và se_texts.csv

Chỉ đánh dấu 1.0 nếu text có nội dung thực sự (không chỉ là tên/tên fallback).

Input:
    data/drug_texts.csv  (idx, drug_name, drug_text)
    data/se_texts.csv    (idx, se_name, se_text)

Output:
    data/drug_text_mask.pt  (757,)  – 1.0 = có mô tả, 0.0 = chỉ có tên
    data/se_text_mask.pt    (994,)  – 1.0 = có mô tả, 0.0 = chỉ có tên
"""

import os
import pandas as pd
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")


def create_drug_mask():
    df = pd.read_csv(os.path.join(DATA, "drug_texts.csv"))
    print(f"Drug texts: {len(df)} rows")

    masks = []
    for _, row in df.iterrows():
        name = str(row["drug_name"]).strip()
        text = str(row["drug_text"]).strip()

        # Heuristic: nếu text dài hơn tên + 10 ký tự → có nội dung thực
        has_content = 1.0 if len(text) > len(name) + 10 else 0.0
        masks.append(has_content)

    mask = torch.tensor(masks, dtype=torch.float32)
    print(f"  Drugs with real text: {mask.sum():.0f}/{len(mask)} ({mask.mean()*100:.1f}%)")
    print(f"  Drugs with only name: {(1-mask).sum():.0f}/{len(mask)}")

    output_path = os.path.join(DATA, "drug_text_mask.pt")
    torch.save(mask, output_path)
    print(f"Saved -> drug_text_mask.pt")
    return mask


def create_se_mask():
    df = pd.read_csv(os.path.join(DATA, "se_texts.csv"))
    print(f"\nSE texts: {len(df)} rows")

    masks = []
    for _, row in df.iterrows():
        name = str(row["se_name"]).strip()
        text = str(row["se_text"]).strip()

        # Heuristic: nếu text dài hơn tên + 5 ký tự → có nội dung thực
        # (SE name thường ngắn hơn drug name)
        has_content = 1.0 if len(text) > len(name) + 5 else 0.0
        masks.append(has_content)

    mask = torch.tensor(masks, dtype=torch.float32)
    print(f"  SEs with real text: {mask.sum():.0f}/{len(mask)} ({mask.mean()*100:.1f}%)")
    print(f"  SEs with only name: {(1-mask).sum():.0f}/{len(mask)}")

    output_path = os.path.join(DATA, "se_text_mask.pt")
    torch.save(mask, output_path)
    print(f"Saved -> se_text_mask.pt")
    return mask


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    create_drug_mask()
    create_se_mask()
    print("\n=== Done! ===")
