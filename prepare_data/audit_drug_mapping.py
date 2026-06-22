"""
Audit the impact of a potentially wrong drug_mapping order on the LLM branch.

Outputs:
    data/drug_mapping_audit/report.md
    data/drug_mapping_audit/summary.json
    data/drug_mapping_audit/drug_names_only_in_frequency_txt.csv
    data/drug_mapping_audit/drug_names_only_in_mapping.csv
    data/drug_mapping_audit/potential_alias_matches.csv
    data/drug_mapping_audit/order_mismatches.csv
    data/drug_mapping_audit/index_samples.csv

Run:
    .\\venv\\Scripts\\python.exe prepare_data/audit_drug_mapping.py
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - optional in audit mode
    torch = None


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = DATA / "drug_mapping_audit"

TXT_PATH = BASE / "drug_names_from_FrequencyData.txt"
MAPPING_PATH = DATA / "drug_mapping.csv"
DRUG_TEXTS_PATH = DATA / "drug_texts.csv"
DRUGBANK_TEXT_PATH = DATA / "drugbank_text.csv"
LLM_PATH = DATA / "drug_llm_features.pt"
MSSF_PATH = BASE / "mssf.py"

SALT_TOKENS = {
    "acetate",
    "besylate",
    "bromide",
    "calcium",
    "chloride",
    "citrate",
    "disodium",
    "ethanolate",
    "ethanolate",
    "fumarate",
    "hcl",
    "hydrochloride",
    "hydrate",
    "hydrobromide",
    "iodide",
    "isethionate",
    "maleate",
    "mesylate",
    "monohydrate",
    "nitrate",
    "phosphate",
    "potassium",
    "sodium",
    "succinate",
    "sulfate",
    "tartrate",
    "tosylate",
}

FORM_TOKENS = {
    "cream",
    "gel",
    "implant",
    "injection",
    "kit",
    "lotion",
    "ointment",
    "patch",
    "powder",
    "solution",
    "spray",
    "tablet",
    "topical",
    "vaccine",
}

CHEMICAL_EQUIV = {
    "acid": "",
    "alendronic": "alendron",
    "azelaic": "azela",
    "clodronic": "clodron",
    "ibandronic": "ibandron",
    "pamidronic": "pamidron",
    "zoledronic": "zoledron",
    "zoledronate": "zoledron",
}


@dataclass
class NameRow:
    idx: int
    raw_name: str
    normalized: str
    core: str
    stemmed_core: str


def strip_prefix_number(text: str) -> str:
    return re.sub(r"^\s*\d+\.\s*", "", text).strip()


def normalize_name(text: str) -> str:
    text = strip_prefix_number(text)
    text = text.casefold()
    text = re.sub(r"[\(\)\[\],;/]+", " ", text)
    text = re.sub(r"[^0-9a-zA-Z\-\+\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return [tok for tok in normalize_name(text).split() if tok]


def stem_token(token: str) -> str:
    if token in CHEMICAL_EQUIV:
        return CHEMICAL_EQUIV[token]
    if token.endswith("ic") and len(token) > 4:
        return token[:-2]
    if token.endswith("ate") and len(token) > 5:
        return token[:-3]
    if token.endswith("ide") and len(token) > 5:
        return token[:-3]
    return token


def core_name(text: str) -> str:
    tokens = [
        tok
        for tok in tokenize(text)
        if tok not in SALT_TOKENS and tok not in FORM_TOKENS
    ]
    return " ".join(tokens)


def stemmed_core_name(text: str) -> str:
    stemmed = [stem_token(tok) for tok in core_name(text).split()]
    stemmed = [tok for tok in stemmed if tok]
    return " ".join(stemmed)


def sequence_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def read_text_names(path: Path) -> List[NameRow]:
    rows: List[NameRow] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for idx, line in enumerate(handle):
            raw = line.strip()
            if not raw:
                continue
            rows.append(
                NameRow(
                    idx=idx,
                    raw_name=strip_prefix_number(raw),
                    normalized=normalize_name(raw),
                    core=core_name(raw),
                    stemmed_core=stemmed_core_name(raw),
                )
            )
    return rows


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_mapping_names(path: Path) -> List[NameRow]:
    rows = read_csv_rows(path)
    result: List[NameRow] = []
    for row in rows:
        idx = int(row["idx"])
        name = row["drug_name"].strip()
        result.append(
            NameRow(
                idx=idx,
                raw_name=name,
                normalized=normalize_name(name),
                core=core_name(name),
                stemmed_core=stemmed_core_name(name),
            )
        )
    return result


def best_alias_candidate(
    txt_row: NameRow,
    mapping_candidates: Sequence[NameRow],
) -> Optional[Tuple[NameRow, float, str]]:
    best: Optional[Tuple[NameRow, float, str]] = None
    txt_tokens = set(txt_row.core.split())
    txt_stemmed = set(txt_row.stemmed_core.split())

    for candidate in mapping_candidates:
        map_tokens = set(candidate.core.split())
        map_stemmed = set(candidate.stemmed_core.split())
        shared_tokens = txt_tokens & map_tokens
        shared_stemmed = txt_stemmed & map_stemmed
        overlap = len(shared_tokens) / max(1, min(len(txt_tokens or {""}), len(map_tokens or {""})))
        stemmed_overlap = len(shared_stemmed) / max(
            1, min(len(txt_stemmed or {""}), len(map_stemmed or {""}))
        )
        raw_ratio = sequence_ratio(txt_row.normalized, candidate.normalized)
        core_ratio = sequence_ratio(txt_row.core, candidate.core)
        stem_ratio = sequence_ratio(txt_row.stemmed_core, candidate.stemmed_core)

        reason: Optional[str] = None
        score = max(raw_ratio, core_ratio, stem_ratio)
        if txt_row.core and txt_row.core == candidate.core:
            reason = "same_core_without_salt_or_form"
            score = 1.0
        elif txt_row.stemmed_core and txt_row.stemmed_core == candidate.stemmed_core:
            reason = "same_stemmed_core"
            score = 0.98
        elif (
            score >= 0.92
            and (overlap >= 0.5 or stemmed_overlap >= 0.5)
        ):
            reason = "high_string_similarity"
        elif (
            score >= 0.84
            and (txt_row.core in candidate.core or candidate.core in txt_row.core)
            and (shared_tokens or shared_stemmed)
        ):
            reason = "likely_base_drug_plus_salt_or_form"
        elif stemmed_overlap >= 1.0 and score >= 0.75:
            reason = "full_stemmed_token_overlap"

        if reason is None:
            continue
        if best is None or score > best[1]:
            best = (candidate, score, reason)
    return best


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_expected_drug_text(drug_name: str, description: str, moa: str) -> str:
    parts: List[str] = []
    description = (description or "").strip()
    moa = (moa or "").strip()
    if description:
        parts.append(description[:800])
    if moa:
        parts.append(moa[:400])
    return " ".join(parts) if parts else drug_name


def inspect_use_llm_default(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8")
    has_flag = "--use_llm" in text
    default_false = bool(
        re.search(
            r"add_argument\(\s*['\"]--use_llm['\"].*?default\s*=\s*False",
            text,
            flags=re.S,
        )
    )
    gated_load = "drug_llm_features.pt" in text and "if getattr(args, 'use_llm', False)" in text
    return {
        "has_use_llm_flag": has_flag,
        "use_llm_default_false": default_false,
        "llm_loading_gated_by_flag": gated_load,
    }


def sample_indices(total: int) -> List[int]:
    candidates = [0, 1, 2, 10, 50, 100, 250, 500, total - 1]
    seen = []
    for idx in candidates:
        if 0 <= idx < total and idx not in seen:
            seen.append(idx)
    return seen


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    txt_rows = read_text_names(TXT_PATH)
    mapping_csv_rows = read_csv_rows(MAPPING_PATH)
    mapping_rows = read_mapping_names(MAPPING_PATH)
    drug_text_rows = read_csv_rows(DRUG_TEXTS_PATH)
    drugbank_rows = read_csv_rows(DRUGBANK_TEXT_PATH)

    use_llm_info = inspect_use_llm_default(MSSF_PATH)

    txt_by_norm = {row.normalized: row for row in txt_rows}
    mapping_by_norm = {row.normalized: row for row in mapping_rows}

    txt_norms = set(txt_by_norm)
    mapping_norms = set(mapping_by_norm)
    exact_overlap_norms = txt_norms & mapping_norms

    order_matches = 0
    order_mismatches: List[Dict[str, object]] = []
    for norm_name in sorted(exact_overlap_norms):
        txt_row = txt_by_norm[norm_name]
        mapping_row = mapping_by_norm[norm_name]
        if txt_row.idx == mapping_row.idx:
            order_matches += 1
        else:
            order_mismatches.append(
                {
                    "drug_name": mapping_row.raw_name,
                    "txt_idx": txt_row.idx,
                    "mapping_idx": mapping_row.idx,
                    "delta": mapping_row.idx - txt_row.idx,
                }
            )

    txt_only_rows = [txt_by_norm[name] for name in sorted(txt_norms - mapping_norms)]
    mapping_only_rows = [mapping_by_norm[name] for name in sorted(mapping_norms - txt_norms)]

    unmatched_mapping = mapping_only_rows.copy()
    potential_alias_matches: List[Dict[str, object]] = []
    matched_mapping_norms = set()
    remaining_txt_only: List[NameRow] = []

    for txt_row in txt_only_rows:
        candidates = [row for row in unmatched_mapping if row.normalized not in matched_mapping_norms]
        best = best_alias_candidate(txt_row, candidates)
        if best is None:
            remaining_txt_only.append(txt_row)
            continue

        mapping_row, score, reason = best
        matched_mapping_norms.add(mapping_row.normalized)
        potential_alias_matches.append(
            {
                "txt_idx": txt_row.idx,
                "txt_name": txt_row.raw_name,
                "mapping_idx": mapping_row.idx,
                "mapping_name": mapping_row.raw_name,
                "score": f"{score:.4f}",
                "reason": reason,
            }
        )

    remaining_mapping_only = [
        row for row in unmatched_mapping if row.normalized not in matched_mapping_norms
    ]

    drug_texts_by_idx = {
        int(row["idx"]): {
            "drug_name": row["drug_name"].strip(),
            "drug_text": row["drug_text"],
        }
        for row in drug_text_rows
    }
    drugbank_by_id = {
        row["drugbank_id"].strip(): {
            "drug_name": row["drug_name"].strip(),
            "description": row.get("description", ""),
            "moa": row.get("moa", ""),
        }
        for row in drugbank_rows
    }

    mapping_to_drugtexts_name_mismatches = 0
    mapping_to_expected_text_mismatches = 0
    for row in mapping_rows:
        actual = drug_texts_by_idx.get(row.idx)
        if actual is None or actual["drug_name"] != row.raw_name:
            mapping_to_drugtexts_name_mismatches += 1
        expected_source = drugbank_by_id.get(mapping_csv_rows[row.idx]["drugbank_id"].strip())
        if expected_source is None or actual is None:
            continue
        expected_text = make_expected_drug_text(
            row.raw_name,
            expected_source.get("description", ""),
            expected_source.get("moa", ""),
        )
        if actual["drug_text"] != expected_text:
            mapping_to_expected_text_mismatches += 1

    llm_rows = None
    llm_dim = None
    llm_loaded = False
    llm_sample_norms: Dict[int, float] = {}
    if torch is not None and LLM_PATH.exists():
        tensor = torch.load(LLM_PATH, map_location="cpu", weights_only=True)
        llm_loaded = True
        if hasattr(tensor, "shape") and len(tensor.shape) == 2:
            llm_rows = int(tensor.shape[0])
            llm_dim = int(tensor.shape[1])
            for idx in sample_indices(len(mapping_rows)):
                llm_sample_norms[idx] = float(torch.linalg.vector_norm(tensor[idx]).item())

    sample_rows: List[Dict[str, object]] = []
    for idx in sample_indices(len(mapping_rows)):
        mapping_row = mapping_rows[idx]
        txt_same_idx = txt_rows[idx] if idx < len(txt_rows) else None
        current_text_row = drug_texts_by_idx.get(idx)
        txt_position = txt_by_norm.get(mapping_row.normalized)
        llm_norm = llm_sample_norms.get(idx)
        sample_rows.append(
            {
                "idx": idx,
                "txt_name_at_same_idx": txt_same_idx.raw_name if txt_same_idx else "",
                "mapping_drug_name": mapping_row.raw_name,
                "mapping_drug_name_position_in_txt": txt_position.idx if txt_position else "",
                "drug_text_name": current_text_row["drug_name"] if current_text_row else "",
                "drug_text_prefix": (
                    current_text_row["drug_text"][:120].replace("\n", " ")
                    if current_text_row
                    else ""
                ),
                "llm_vector_norm": f"{llm_norm:.6f}" if llm_norm is not None else "",
            }
        )

    write_csv(
        OUT / "drug_names_only_in_frequency_txt.csv",
        ["txt_idx", "txt_name", "normalized_name", "core_name", "stemmed_core"],
        [
            {
                "txt_idx": row.idx,
                "txt_name": row.raw_name,
                "normalized_name": row.normalized,
                "core_name": row.core,
                "stemmed_core": row.stemmed_core,
            }
            for row in remaining_txt_only
        ],
    )
    write_csv(
        OUT / "drug_names_only_in_mapping.csv",
        ["mapping_idx", "drugbank_id", "mapping_name", "normalized_name", "core_name", "stemmed_core"],
        [
            {
                "mapping_idx": row.idx,
                "drugbank_id": mapping_csv_rows[row.idx]["drugbank_id"].strip(),
                "mapping_name": row.raw_name,
                "normalized_name": row.normalized,
                "core_name": row.core,
                "stemmed_core": row.stemmed_core,
            }
            for row in remaining_mapping_only
        ],
    )
    write_csv(
        OUT / "potential_alias_matches.csv",
        ["txt_idx", "txt_name", "mapping_idx", "mapping_name", "score", "reason"],
        potential_alias_matches,
    )
    write_csv(
        OUT / "order_mismatches.csv",
        ["drug_name", "txt_idx", "mapping_idx", "delta"],
        order_mismatches,
    )
    write_csv(
        OUT / "index_samples.csv",
        [
            "idx",
            "txt_name_at_same_idx",
            "mapping_drug_name",
            "mapping_drug_name_position_in_txt",
            "drug_text_name",
            "drug_text_prefix",
            "llm_vector_norm",
        ],
        sample_rows,
    )

    summary = {
        "txt_drug_count": len(txt_rows),
        "mapping_drug_count": len(mapping_rows),
        "exact_name_overlap_after_normalization": len(exact_overlap_norms),
        "order_matches_on_exact_overlap": order_matches,
        "order_mismatches_on_exact_overlap": len(order_mismatches),
        "txt_only_after_exact_normalization": len(txt_only_rows),
        "mapping_only_after_exact_normalization": len(mapping_only_rows),
        "potential_alias_matches": len(potential_alias_matches),
        "remaining_txt_only_after_alias_heuristic": len(remaining_txt_only),
        "remaining_mapping_only_after_alias_heuristic": len(remaining_mapping_only),
        "use_llm_flag_present": use_llm_info["has_use_llm_flag"],
        "use_llm_default_false": use_llm_info["use_llm_default_false"],
        "llm_loading_gated_by_flag": use_llm_info["llm_loading_gated_by_flag"],
        "drug_text_rows": len(drug_text_rows),
        "mapping_to_drugtexts_name_mismatches": mapping_to_drugtexts_name_mismatches,
        "mapping_to_expected_text_mismatches": mapping_to_expected_text_mismatches,
        "llm_tensor_loaded": llm_loaded,
        "llm_tensor_rows": llm_rows,
        "llm_tensor_dim": llm_dim,
        "llm_row_count_matches_mapping": llm_rows == len(mapping_rows) if llm_rows is not None else None,
    }

    if use_llm_info["has_use_llm_flag"] and use_llm_info["llm_loading_gated_by_flag"]:
        if len(order_mismatches) > 0 or len(remaining_txt_only) > 0 or len(remaining_mapping_only) > 0:
            summary["risk_level"] = "blocker_if_use_llm_enabled"
            summary["conclusion"] = (
                "Current LLM/text artifacts are internally consistent with drug_mapping.csv, "
                "but they are unsafe for the original MSSF drug index if drug_mapping.csv is wrong."
            )
        else:
            summary["risk_level"] = "low"
            summary["conclusion"] = (
                "No material order/name mismatch was found after normalization."
            )
    else:
        summary["risk_level"] = "name_interpretation_risk"
        summary["conclusion"] = (
            "Base MSSF can still run without LLM, but any analysis by drug name may be wrong if mapping.csv is wrong."
        )

    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    order_mismatch_examples = order_mismatches[:5]
    alias_examples = potential_alias_matches[:5]
    txt_only_examples = remaining_txt_only[:5]
    mapping_only_examples = remaining_mapping_only[:5]

    report_lines = [
        "# Drug Mapping Audit",
        "",
        "## Summary",
        f"- `mssf.py` exposes `--use_llm`: `{use_llm_info['has_use_llm_flag']}`",
        f"- `--use_llm` default is `False`: `{use_llm_info['use_llm_default_false']}`",
        f"- LLM loading is gated by the flag: `{use_llm_info['llm_loading_gated_by_flag']}`",
        f"- `drug_names_from_FrequencyData.txt`: `{len(txt_rows)}` names",
        f"- `data/drug_mapping.csv`: `{len(mapping_rows)}` names",
        f"- Exact overlap after normalization: `{len(exact_overlap_norms)}`",
        f"- Same index on exact overlap: `{order_matches}`",
        f"- Different index on exact overlap: `{len(order_mismatches)}`",
        f"- Potential alias/salt/form matches: `{len(potential_alias_matches)}`",
        f"- Remaining txt-only names: `{len(remaining_txt_only)}`",
        f"- Remaining mapping-only names: `{len(remaining_mapping_only)}`",
        f"- `data/drug_texts.csv` rows: `{len(drug_text_rows)}`",
        f"- Mapping -> `drug_texts.csv` name mismatches: `{mapping_to_drugtexts_name_mismatches}`",
        f"- Mapping -> expected text mismatches: `{mapping_to_expected_text_mismatches}`",
        f"- `drug_llm_features.pt` loaded: `{llm_loaded}`",
    ]

    if llm_rows is not None and llm_dim is not None:
        report_lines.extend(
            [
                f"- `drug_llm_features.pt` shape: `({llm_rows}, {llm_dim})`",
                f"- LLM row count matches mapping: `{llm_rows == len(mapping_rows)}`",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Conclusion",
            f"- Risk level: `{summary['risk_level']}`",
            f"- {summary['conclusion']}",
            "- This means the current text and LLM feature files follow the current `drug_mapping.csv` order, so if that mapping is wrong relative to the original 757-drug matrix, the LLM branch will attach the wrong text feature to the wrong drug index.",
            "",
            "## Example Order Mismatches",
        ]
    )
    if order_mismatch_examples:
        for row in order_mismatch_examples:
            report_lines.append(
                f"- `{row['drug_name']}`: txt idx `{row['txt_idx']}`, mapping idx `{row['mapping_idx']}`"
            )
    else:
        report_lines.append("- None")

    report_lines.extend(["", "## Example Potential Alias Matches"])
    if alias_examples:
        for row in alias_examples:
            report_lines.append(
                f"- `{row['txt_name']}` -> `{row['mapping_name']}` (`{row['reason']}`, score `{row['score']}`)"
            )
    else:
        report_lines.append("- None")

    report_lines.extend(["", "## Example Remaining Txt-Only Names"])
    if txt_only_examples:
        for row in txt_only_examples:
            report_lines.append(f"- `{row.raw_name}`")
    else:
        report_lines.append("- None")

    report_lines.extend(["", "## Example Remaining Mapping-Only Names"])
    if mapping_only_examples:
        for row in mapping_only_examples:
            report_lines.append(f"- `{row.raw_name}`")
    else:
        report_lines.append("- None")

    report_lines.extend(
        [
            "",
            "## Output Files",
            "- `data/drug_mapping_audit/summary.json`",
            "- `data/drug_mapping_audit/order_mismatches.csv`",
            "- `data/drug_mapping_audit/potential_alias_matches.csv`",
            "- `data/drug_mapping_audit/drug_names_only_in_frequency_txt.csv`",
            "- `data/drug_mapping_audit/drug_names_only_in_mapping.csv`",
            "- `data/drug_mapping_audit/index_samples.csv`",
        ]
    )

    (OUT / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("Saved audit outputs to: data/drug_mapping_audit")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
