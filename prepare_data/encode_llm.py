"""
encode_llm.py - Encode drug/SE texts bang PubMedBERT -> .pt files

Input:
    data/drug_texts.csv  (idx, drug_name, drug_text)
    data/se_texts.csv    (idx, se_name, se_text)

Output:
    data/drug_llm_features.pt  shape (N_drugs, 768)
    data/se_llm_features.pt    shape (N_side_effects, 768)

Chay:
    python prepare_data/encode_llm.py
    python prepare_data/encode_llm.py --drug-texts data/benchmark_drug_texts.csv --se-texts data/benchmark_se_texts.csv
"""

import argparse
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

BASE = r"d:\Duc\Do an thac si\MSSF\MSSF"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


def parse_args():
    parser = argparse.ArgumentParser(description="Encode drug/SE texts with PubMedBERT.")
    parser.add_argument("--drug-texts", default=os.path.join(DATA, "drug_texts.csv"),
                        help="Input drug texts CSV.")
    parser.add_argument("--se-texts", default=os.path.join(DATA, "se_texts.csv"),
                        help="Input side-effect texts CSV.")
    parser.add_argument("--drug-output", default=os.path.join(DATA, "drug_llm_features.pt"),
                        help="Output path for drug LLM features.")
    parser.add_argument("--se-output", default=os.path.join(DATA, "se_llm_features.pt"),
                        help="Output path for side-effect LLM features.")
    return parser.parse_args()


tokenizer = None
model = None
device = None


def load_encoder():
    global tokenizer, model, device
    print(f"Loading PubMedBERT: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")

def encode_single(text, max_length=512, stride=256):
    """Encode má»™t text dÃ i báº±ng sliding window chunking + mean pooling.
    
    Tokenize toÃ n bá»™ text, chia thÃ nh cÃ¡c chunk 512 token (overlap stride token),
    encode tá»«ng chunk láº¥y CLS embedding, rá»“i average táº¥t cáº£ chunks láº¡i.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=False,
                       add_special_tokens=False)
    input_ids = tokens["input_ids"][0]  # (N_tokens,)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    chunk_embeddings = []
    # Má»—i chunk thá»±c sá»± chá»©a max_length-2 token ná»™i dung (trá»« [CLS] vÃ  [SEP])
    content_len = max_length - 2
    pos = 0
    while pos < len(input_ids):
        chunk = input_ids[pos : pos + content_len]
        # ThÃªm [CLS] Ä‘áº§u vÃ  [SEP] cuá»‘i
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
        pos += content_len - stride  # bÆ°á»›c nháº£y = content_len - overlap

    return torch.stack(chunk_embeddings).mean(0)  # (1, 768)


def encode_texts(texts, max_length=512, stride=256):
    """Encode list of texts -> (N, 768) tensor dÃ¹ng chunking + mean pooling."""
    all_embeddings = []
    for i, text in enumerate(texts):
        emb = encode_single(text, max_length=max_length, stride=stride)
        all_embeddings.append(emb)
        if (i + 1) % 50 == 0 or (i + 1) == len(texts):
            print(f"  Encoded {i + 1}/{len(texts)}")
    return torch.cat(all_embeddings, dim=0)

def main():
    args = parse_args()
    load_encoder()

    # -- Drug texts ------------------------------------------------------------
    drug_df = pd.read_csv(args.drug_texts)
    drug_texts = drug_df["drug_text"].fillna("").tolist()

    print(f"\nEncoding {len(drug_texts)} drug texts from {args.drug_texts}...")
    drug_vecs = encode_texts(drug_texts)
    print(f"Drug vectors: {drug_vecs.shape}")
    torch.save(drug_vecs, args.drug_output)
    print(f"Saved -> {args.drug_output}")

    # -- SE texts --------------------------------------------------------------
    se_df = pd.read_csv(args.se_texts)
    se_texts = se_df["se_text"].fillna("").tolist()

    print(f"\nEncoding {len(se_texts)} SE texts from {args.se_texts}...")
    se_vecs = encode_texts(se_texts)
    print(f"SE vectors: {se_vecs.shape}")
    torch.save(se_vecs, args.se_output)
    print(f"Saved -> {args.se_output}")

    print("\n=== Done! ===")
    print(f"drug_llm_features: {drug_vecs.shape}")
    print(f"se_llm_features  : {se_vecs.shape}")


if __name__ == "__main__":
    main()





