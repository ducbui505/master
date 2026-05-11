"""
encode_llm.py - Encode drug/SE texts bang PubMedBERT -> .pt files

Input:
    data/drug_texts.csv  (idx, drug_name, drug_text)
    data/se_texts.csv    (idx, se_name, se_text)

Output:
    data/drug_llm_features.pt  shape (757, 768)
    data/se_llm_features.pt    shape (994, 768)

Chay:
    python prepare_data/encode_llm.py
"""

import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

BASE = r"d:\Duc\Do an thac si\MSSF\MSSF"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
print(f"Loading PubMedBERT: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
print(f"Device: {device}")

def encode_texts(texts, batch_size=32, max_length=256):
    """Encode list of texts -> (N, 768) tensor using CLS token."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            out = model(**encoded)
            cls = out.last_hidden_state[:, 0, :]  # (batch, 768)
        all_embeddings.append(cls.cpu())
        print(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)}")
    return torch.cat(all_embeddings, dim=0)

# -- Drug texts ----------------------------------------------------------------
drug_df    = pd.read_csv(os.path.join(DATA, "drug_texts.csv"))
drug_texts = drug_df["drug_text"].fillna("").tolist()

print(f"\nEncoding {len(drug_texts)} drug texts...")
drug_vecs = encode_texts(drug_texts)
print(f"Drug vectors: {drug_vecs.shape}")
torch.save(drug_vecs, os.path.join(DATA, "drug_llm_features.pt"))
print("Saved -> data/drug_llm_features.pt")

# -- SE texts ------------------------------------------------------------------
se_df    = pd.read_csv(os.path.join(DATA, "se_texts.csv"))
se_texts = se_df["se_text"].fillna("").tolist()

print(f"\nEncoding {len(se_texts)} SE texts...")
se_vecs = encode_texts(se_texts)
print(f"SE vectors: {se_vecs.shape}")
torch.save(se_vecs, os.path.join(DATA, "se_llm_features.pt"))
print("Saved -> data/se_llm_features.pt")

print("\n=== Done! ===")
print(f"drug_llm_features: {drug_vecs.shape}")
print(f"se_llm_features  : {se_vecs.shape}")
