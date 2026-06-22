"""
Build canonical drug/SE mappings from benchmark ordered lists.

This script does not overwrite the legacy MSSF mappings because the current
workspace raw matrix is 757x994 while the benchmark reference is 759x994.

Outputs:
    data/benchmark_drug_mapping.csv
    data/benchmark_se_mapping.csv
    data/benchmark_drug_mapping_unresolved.csv
    data/benchmark_mapping_report.md
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"

BENCH_DRUGS = DATA / "benchmark_drugs_ordered.txt"
BENCH_SES = DATA / "benchmark_sideeffects_ordered.txt"
BENCH_SUMMARY = DATA / "benchmark_axes_summary.txt"

DRUGBANK_TEXT = DATA / "drugbank_text.csv"
SIDER_ALL_DRUGS = DATA / "sider_all_drugs.csv"
SIDER_ALL_SES = DATA / "sider_all_ses.csv"

OUT_DRUG = DATA / "benchmark_drug_mapping.csv"
OUT_SE = DATA / "benchmark_se_mapping.csv"
OUT_UNRESOLVED = DATA / "benchmark_drug_mapping_unresolved.csv"
OUT_REPORT = DATA / "benchmark_mapping_report.md"


SALT_TOKENS = {
    "acetate",
    "besylate",
    "bromide",
    "calcium",
    "chloride",
    "citrate",
    "disodium",
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

MANUAL_ALIASES: Dict[str, Tuple[str, str, str]] = {
    "albumin": ("DB00062", "Albumin human", "manual_generic_to_human_albumin"),
    "beclometasone": ("DB00394", "Beclomethasone", "manual_uk_us_spelling"),
    "calcium folinate": ("DB00650", "Leucovorin", "manual_synonym_folinic_acid"),
    "chlormethine": ("DB00888", "Mechlorethamine", "manual_uk_us_spelling"),
    "chlortalidone": ("DB00310", "Chlorthalidone", "manual_uk_us_spelling"),
    "ciclosporin": ("DB00091", "Cyclosporine", "manual_uk_us_spelling"),
    "colestilan": ("", "", "unresolved_missing_from_local_sources"),
    "dicycloverine": ("DB00804", "Dicyclomine", "manual_uk_us_spelling"),
    "fampridine": ("DB06637", "Dalfampridine", "manual_inn_usan_synonym"),
    "florbetapir (18f)": ("DB09149", "Florbetapir F-18", "manual_radiopharmaceutical_name_variant"),
    "glutamine": ("DB00130", "L-Glutamine", "manual_generic_to_l_form"),
    "glyceryl trinitrate": ("DB00727", "Nitroglycerin", "manual_synonym"),
    "hydroxycarbamide": ("DB01005", "Hydroxyurea", "manual_uk_us_spelling"),
    "ibandronic acid": ("DB00710", "Ibandronate", "manual_acid_to_drugbank_name"),
    "indometacin": ("DB00328", "Indomethacin", "manual_uk_us_spelling"),
    "iodine ioflupane (123i)": ("DB08824", "Ioflupane I 123", "manual_radiopharmaceutical_name_variant"),
    "iodine ioflupane 123i": ("DB08824", "Ioflupane I 123", "manual_radiopharmaceutical_name_variant"),
    "lamivudine and abacavir": ("", "", "unresolved_combo_missing_from_local_sources"),
    "leuprorelin": ("DB00007", "Leuprolide", "manual_inn_usan_synonym"),
    "mercaptamine": ("DB00847", "Cysteamine", "manual_synonym"),
    "mesna": ("DB09110", "Coenzyme M", "manual_common_name_to_drugbank_entry"),
    "muromonab-cd3": ("DB00075", "Muromonab", "manual_antibody_name_variant"),
    "olmesartan medoxomil": ("DB00275", "Olmesartan", "manual_prodrug_to_active_drug_entry"),
    "paracetamol": ("DB00316", "Acetaminophen", "manual_inn_usan_synonym"),
    "pentamidine isethionate": ("DB00738", "Pentamidine", "manual_strip_salt"),
    "podophyllotoxin": ("DB01179", "Podofilox", "manual_synonym"),
    "radium (223ra) dichloride": ("DB08913", "Radium Ra 223 Dichloride", "manual_radiopharmaceutical_name_variant"),
    "radium 223ra dichloride": ("DB08913", "Radium Ra 223 Dichloride", "manual_radiopharmaceutical_name_variant"),
    "retigabine": ("DB04953", "Ezogabine", "manual_inn_usan_synonym"),
    "saccharated iron oxide": ("DB09146", "Iron sucrose", "manual_synonym_iron_saccharate"),
    "salmeterol and fluticasone": ("", "", "unresolved_combo_missing_from_local_sources"),
    "secretin": ("DB09532", "Secretin human", "manual_generic_to_human_secretin"),
    "sodium phosphate": ("", "", "unresolved_ambiguous_multiple_forms"),
    "sodium phenylbutyrate": ("DB06819", "Phenylbutyric acid", "manual_salt_to_active_acid_entry"),
    "sulfamethoxazole and trimethoprim": ("", "", "unresolved_combo_missing_from_local_sources"),
    "technetium (99mtc) exametazime": ("DB09163", "Technetium Tc-99m exametazime", "manual_radiopharmaceutical_name_variant"),
    "technetium (99mtc) tetrofosmin": ("DB09160", "Technetium Tc-99m tetrofosmin", "manual_radiopharmaceutical_name_variant"),
    "tiotropium bromide": ("DB01409", "Tiotropium", "manual_strip_salt"),
    "umeclidinium bromide": ("DB09076", "Umeclidinium", "manual_strip_salt"),
    "cyproterone": ("DB04839", "Cyproterone acetate", "manual_base_to_ester"),
}


@dataclass
class Candidate:
    drugbank_id: str
    name: str
    source: str
    normalized: str
    core: str
    stemmed: str


def strip_prefix_number(text: str) -> str:
    return re.sub(r"^\s*\d+\.\s*", "", text).strip()


def normalize_name(text: str) -> str:
    text = strip_prefix_number(text)
    text = text.casefold().strip()
    text = re.sub(r"[\(\)\[\],;/]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def core_name(text: str) -> str:
    tokens = [
        token
        for token in normalize_name(text).split()
        if token not in SALT_TOKENS and token not in FORM_TOKENS
    ]
    return " ".join(tokens)


def stem_token(token: str) -> str:
    replacements = {
        "acid": "",
        "alendronic": "alendron",
        "beclometasone": "beclomethasone",
        "chlortalidone": "chlorthalidone",
        "clodronic": "clodron",
        "ciclosporin": "cyclosporine",
        "cyproterone": "cyproterone",
        "dicycloverine": "dicyclomine",
        "fampridine": "dalfampridine",
        "hydroxycarbamide": "hydroxyurea",
        "ibandronic": "ibandron",
        "indometacin": "indomethacin",
        "leuprorelin": "leuprolide",
        "mercaptamine": "cysteamine",
        "paracetamol": "acetaminophen",
        "retigabine": "ezogabine",
        "zoledronic": "zoledron",
        "zoledronate": "zoledron",
    }
    if token in replacements:
        return replacements[token]
    return token


def stemmed_core_name(text: str) -> str:
    tokens = [stem_token(token) for token in core_name(text).split()]
    return " ".join(token for token in tokens if token)


def read_txt_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [strip_prefix_number(line.strip()) for line in handle if line.strip()]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_candidates() -> List[Candidate]:
    candidates: List[Candidate] = []
    seen = set()
    for row in read_csv_rows(DRUGBANK_TEXT):
        key = (row["drugbank_id"].strip(), row["drug_name"].strip())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            Candidate(
                drugbank_id=key[0],
                name=key[1],
                source="drugbank_text",
                normalized=normalize_name(key[1]),
                core=core_name(key[1]),
                stemmed=stemmed_core_name(key[1]),
            )
        )
    for row in read_csv_rows(SIDER_ALL_DRUGS):
        key = (row["drugbank_id"].strip(), row["drugbank_name"].strip())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            Candidate(
                drugbank_id=key[0],
                name=key[1],
                source="sider_all_drugs",
                normalized=normalize_name(key[1]),
                core=core_name(key[1]),
                stemmed=stemmed_core_name(key[1]),
            )
        )
    return candidates


def build_candidate_indexes(
    candidates: Sequence[Candidate],
) -> Tuple[Dict[str, Candidate], Dict[str, Candidate], Dict[str, Candidate]]:
    norm_map: Dict[str, Candidate] = {}
    core_map: Dict[str, Candidate] = {}
    stem_map: Dict[str, Candidate] = {}
    for candidate in candidates:
        norm_map.setdefault(candidate.normalized, candidate)
        if candidate.core:
            core_map.setdefault(candidate.core, candidate)
        if candidate.stemmed:
            stem_map.setdefault(candidate.stemmed, candidate)
    return norm_map, core_map, stem_map


def fuzzy_candidate(name: str, candidates: Sequence[Candidate]) -> Optional[Tuple[Candidate, float]]:
    normalized = normalize_name(name)
    core = core_name(name)
    stemmed = stemmed_core_name(name)
    scored: List[Tuple[float, Candidate]] = []
    for candidate in candidates:
        score = max(
            SequenceMatcher(None, normalized, candidate.normalized).ratio(),
            SequenceMatcher(None, core, candidate.core).ratio() if core and candidate.core else 0.0,
            SequenceMatcher(None, stemmed, candidate.stemmed).ratio() if stemmed and candidate.stemmed else 0.0,
        )
        if score >= 0.90:
            scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return None
    if len(scored) == 1:
        return scored[0][1], scored[0][0]
    if scored[0][0] - scored[1][0] >= 0.03:
        return scored[0][1], scored[0][0]
    return None


def resolve_drug(
    name: str,
    candidates: Sequence[Candidate],
    norm_map: Dict[str, Candidate],
    core_map: Dict[str, Candidate],
    stem_map: Dict[str, Candidate],
) -> Dict[str, object]:
    normalized = normalize_name(name)
    if normalized in norm_map:
        candidate = norm_map[normalized]
        return {
            "drugbank_id": candidate.drugbank_id,
            "matched_name": candidate.name,
            "resolution_method": f"exact_{candidate.source}",
        }

    if normalized in MANUAL_ALIASES:
        drugbank_id, matched_name, method = MANUAL_ALIASES[normalized]
        return {
            "drugbank_id": drugbank_id,
            "matched_name": matched_name,
            "resolution_method": method,
        }

    core = core_name(name)
    if core and core in core_map:
        candidate = core_map[core]
        return {
            "drugbank_id": candidate.drugbank_id,
            "matched_name": candidate.name,
            "resolution_method": f"core_match_{candidate.source}",
        }

    stemmed = stemmed_core_name(name)
    if stemmed and stemmed in stem_map:
        candidate = stem_map[stemmed]
        return {
            "drugbank_id": candidate.drugbank_id,
            "matched_name": candidate.name,
            "resolution_method": f"stemmed_match_{candidate.source}",
        }

    fuzzy = fuzzy_candidate(name, candidates)
    if fuzzy is not None:
        candidate, score = fuzzy
        return {
            "drugbank_id": candidate.drugbank_id,
            "matched_name": candidate.name,
            "resolution_method": f"fuzzy_{candidate.source}_{score:.3f}",
        }

    return {
        "drugbank_id": "",
        "matched_name": "",
        "resolution_method": "unresolved_no_local_match",
    }


def build_sideeffect_map() -> List[Dict[str, object]]:
    benchmark_ses = read_txt_lines(BENCH_SES)
    sider_rows = read_csv_rows(SIDER_ALL_SES)
    lookup = {}
    for row in sider_rows:
        lookup.setdefault(normalize_name(row["se_name"]), row["umls_cui"].strip())

    mapped_rows: List[Dict[str, object]] = []
    for idx, se_name in enumerate(benchmark_ses):
        key = normalize_name(se_name)
        umls = lookup.get(key, "")
        mapped_rows.append(
            {
                "idx": idx,
                "umls_cui": umls,
                "se_name": se_name,
            }
        )
    return mapped_rows


def main() -> None:
    benchmark_drugs = read_txt_lines(BENCH_DRUGS)
    benchmark_ses = read_txt_lines(BENCH_SES)
    benchmark_summary = BENCH_SUMMARY.read_text(encoding="utf-8-sig").strip()

    candidates = load_candidates()
    norm_map, core_map, stem_map = build_candidate_indexes(candidates)

    drug_rows: List[Dict[str, object]] = []
    unresolved_rows: List[Dict[str, object]] = []
    resolution_counts: Dict[str, int] = {}

    for idx, drug_name in enumerate(benchmark_drugs):
        resolved = resolve_drug(drug_name, candidates, norm_map, core_map, stem_map)
        row = {
            "idx": idx,
            "drugbank_id": resolved["drugbank_id"],
            "drug_name": drug_name,
            "matched_name": resolved["matched_name"],
            "resolution_method": resolved["resolution_method"],
        }
        drug_rows.append(row)
        resolution_counts[row["resolution_method"]] = resolution_counts.get(row["resolution_method"], 0) + 1
        if not row["drugbank_id"]:
            unresolved_rows.append(row)

    se_rows = build_sideeffect_map()
    unresolved_ses = [row for row in se_rows if not row["umls_cui"]]

    write_csv(
        OUT_DRUG,
        ["idx", "drugbank_id", "drug_name", "matched_name", "resolution_method"],
        drug_rows,
    )
    write_csv(OUT_SE, ["idx", "umls_cui", "se_name"], se_rows)
    write_csv(
        OUT_UNRESOLVED,
        ["idx", "drug_name", "resolution_method"],
        [
            {
                "idx": row["idx"],
                "drug_name": row["drug_name"],
                "resolution_method": row["resolution_method"],
            }
            for row in unresolved_rows
        ],
    )

    report_lines = [
        "# Benchmark Mapping Report",
        "",
        "## Reference",
        "```text",
        benchmark_summary,
        "```",
        "",
        "## Outputs",
        f"- `{OUT_DRUG.name}`: `{len(drug_rows)}` rows",
        f"- `{OUT_SE.name}`: `{len(se_rows)}` rows",
        f"- `{OUT_UNRESOLVED.name}`: `{len(unresolved_rows)}` unresolved drug IDs",
        "",
        "## Important Note",
        "- Legacy `data/drug_mapping.csv` and `data/se_mapping.csv` were not overwritten.",
        "- The benchmark reference is `759 x 994`, while the current MSSF raw matrix in this workspace is `757 x 994`.",
        "- Using the benchmark drug mapping directly with the current `Datas/drug_side.pkl` would be unsafe without also switching to the 759-row benchmark matrix.",
        "",
        "## Drug Resolution Breakdown",
    ]
    for method, count in sorted(resolution_counts.items(), key=lambda item: (-item[1], item[0])):
        report_lines.append(f"- `{method}`: `{count}`")

    report_lines.extend(
        [
            "",
            "## Unresolved Drugs",
        ]
    )
    if unresolved_rows:
        for row in unresolved_rows:
            report_lines.append(f"- `{row['idx']}`: `{row['drug_name']}` ({row['resolution_method']})")
    else:
        report_lines.append("- None")

    report_lines.extend(
        [
            "",
            "## Side Effects",
            f"- Exact benchmark side effects mapped to UMLS: `{len(se_rows) - len(unresolved_ses)}/{len(se_rows)}`",
        ]
    )
    if unresolved_ses:
        report_lines.append("- Unresolved SE names:")
        for row in unresolved_ses:
            report_lines.append(f"  - `{row['idx']}`: `{row['se_name']}`")

    OUT_REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Saved {OUT_DRUG.name} with {len(drug_rows)} rows")
    print(f"Saved {OUT_SE.name} with {len(se_rows)} rows")
    print(f"Unresolved benchmark drugs: {len(unresolved_rows)}")
    print(f"Unresolved benchmark side effects: {len(unresolved_ses)}")


if __name__ == "__main__":
    main()
