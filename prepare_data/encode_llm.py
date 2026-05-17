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

def encode_single(text, max_length=512, stride=256):
    """Encode một text dài bằng sliding window chunking + mean pooling.
    
    Tokenize toàn bộ text, chia thành các chunk 512 token (overlap stride token),
    encode từng chunk lấy CLS embedding, rồi average tất cả chunks lại.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=False,
                       add_special_tokens=False)
    input_ids = tokens["input_ids"][0]  # (N_tokens,)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    chunk_embeddings = []
    # Mỗi chunk thực sự chứa max_length-2 token nội dung (trừ [CLS] và [SEP])
    content_len = max_length - 2
    pos = 0
    while pos < len(input_ids):
        chunk = input_ids[pos : pos + content_len]
        # Thêm [CLS] đầu và [SEP] cuối
        chunk_ids = torch.cat([
            torch.tensor([cls_id]),
            chunk,
            torch.tensor([sep_id])
        ]).unsqueeze(0).to(device)
        attn = torch.ones_like(chunk_ids)
        with torch.no_grad():
            out = model(input_ids=chunk_ids, attention_mask=attn)
            cls_emb = out.last_hidden_state[:, 0, :]  # (1, 768)
        chunk_embeddings.append(cls_emb.cpu())
        if pos + content_len >= len(input_ids):
            break
        pos += content_len - stride  # bước nhảy = content_len - overlap

    return torch.stack(chunk_embeddings).mean(0)  # (1, 768)


def encode_texts(texts, max_length=512, stride=256):
    """Encode list of texts -> (N, 768) tensor dùng chunking + mean pooling."""
    all_embeddings = []
    for i, text in enumerate(texts):
        emb = encode_single(text, max_length=max_length, stride=stride)
        all_embeddings.append(emb)
        if (i + 1) % 50 == 0 or (i + 1) == len(texts):
            print(f"  Encoded {i + 1}/{len(texts)}")
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
