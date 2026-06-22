# Benchmark Mapping Report

## Reference
```text
Source: data analysis/data/FrequencyData.mat
R dimensions: 759 rows x 994 columns
Drugs count: 759
Side effects count: 994
Order guarantee:
- Line i in benchmark_drugs_ordered.txt corresponds to row i of matrix R.
- Line j in benchmark_sideeffects_ordered.txt corresponds to column j of matrix R.
```

## Outputs
- `benchmark_drug_mapping.csv`: `759` rows
- `benchmark_se_mapping.csv`: `994` rows
- `benchmark_drug_mapping_unresolved.csv`: `5` unresolved drug IDs

## Important Note
- Legacy `data/drug_mapping.csv` and `data/se_mapping.csv` were not overwritten.
- The benchmark reference is `759 x 994`, while the current MSSF raw matrix in this workspace is `757 x 994`.
- Using the benchmark drug mapping directly with the current `Datas/drug_side.pkl` would be unsafe without also switching to the 759-row benchmark matrix.

## Drug Resolution Breakdown
- `exact_drugbank_text`: `711`
- `manual_uk_us_spelling`: `7`
- `core_match_drugbank_text`: `6`
- `manual_inn_usan_synonym`: `4`
- `exact_sider_all_drugs`: `3`
- `manual_strip_salt`: `3`
- `manual_synonym`: `3`
- `unresolved_combo_missing_from_local_sources`: `3`
- `fuzzy_drugbank_text_0.912`: `2`
- `manual_radiopharmaceutical_name_variant`: `2`
- `fuzzy_drugbank_text_0.903`: `1`
- `fuzzy_drugbank_text_0.963`: `1`
- `manual_acid_to_drugbank_name`: `1`
- `manual_antibody_name_variant`: `1`
- `manual_base_to_ester`: `1`
- `manual_common_name_to_drugbank_entry`: `1`
- `manual_generic_to_human_albumin`: `1`
- `manual_generic_to_human_secretin`: `1`
- `manual_generic_to_l_form`: `1`
- `manual_prodrug_to_active_drug_entry`: `1`
- `manual_salt_to_active_acid_entry`: `1`
- `manual_synonym_folinic_acid`: `1`
- `manual_synonym_iron_saccharate`: `1`
- `unresolved_ambiguous_multiple_forms`: `1`
- `unresolved_missing_from_local_sources`: `1`

## Unresolved Drugs
- `148`: `colestilan` (unresolved_missing_from_local_sources)
- `240`: `sodium phosphate` (unresolved_ambiguous_multiple_forms)
- `338`: `lamivudine and abacavir` (unresolved_combo_missing_from_local_sources)
- `501`: `sulfamethoxazole and trimethoprim` (unresolved_combo_missing_from_local_sources)
- `556`: `salmeterol and fluticasone` (unresolved_combo_missing_from_local_sources)

## Side Effects
- Exact benchmark side effects mapped to UMLS: `994/994`
