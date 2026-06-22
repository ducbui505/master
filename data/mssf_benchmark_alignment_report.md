# MSSF to Benchmark Alignment Report

- MSSF matrix: `757 x 994`
- Benchmark matrix: `759 x 994`
- Exact confident drug mappings: `753/757`
- Exact confident SE mappings: `971/994`
- Ambiguous MSSF drug rows: `4`
- Ambiguous MSSF SE columns: `23`

## Definitely missing from MSSF
- Benchmark idx `338`: `lamivudine and abacavir`
- Benchmark idx `556`: `salmeterol and fluticasone`

## Ambiguous drug candidates
- MSSF idx `65`: `94` phenytoin; `379` fosphenytoin
- MSSF idx `544`: `94` phenytoin; `379` fosphenytoin
- MSSF idx `549`: `240` sodium phosphate; `278` sodium bicarbonate
- MSSF idx `739`: `240` sodium phosphate; `278` sodium bicarbonate

## Why ambiguity remains
The ambiguous drug pairs have identical frequency rows in `FrequencyData.mat` after the confident SE alignment, so the frequency matrix alone cannot distinguish their names.


## Auxiliary drug resolution

Resolved the four ambiguous MSSF drug rows using DrugBank metadata and MSSF auxiliary drug features:

- MSSF idx `65` -> benchmark idx `94` `phenytoin`
  - Evidence: DrugBank lists phenytoin with many targets; MSSF `drug_target.pkl` row 65 has many target indicators.
- MSSF idx `544` -> benchmark idx `379` `fosphenytoin`
  - Evidence: DrugBank lists fosphenytoin with one target; MSSF `drug_target.pkl` row 544 has one target indicator.
- MSSF idx `549` -> benchmark idx `278` `sodium bicarbonate`
  - Evidence: paired complement after assigning row 739 to sodium phosphate; row 549 does not show phosphate/phosphonate nearest-neighbor pattern.
- MSSF idx `739` -> benchmark idx `240` `sodium phosphate`
  - Evidence: MSSF fingerprint nearest neighbors include phosphate/phosphonate drugs such as foscarnet, fosfomycin, pamidronic acid, alendronic acid, ibandronic acid, and zoledronic acid.

Final resolved drug mapping: `757/757`.

Definitely missing benchmark drugs remain:

- Benchmark idx `338`: `lamivudine and abacavir`
- Benchmark idx `556`: `salmeterol and fluticasone`


## Auxiliary side-effect resolution

Resolved all 23 ambiguous MSSF side-effect columns with auxiliary semantic evidence from `Datas/side_effect_semantic.pkl` and `Datas/glove_wordEmbedding.pkl`. These 23 columns are marked as `resolved_by_auxiliary_semantic_features` in `data/mssf_to_benchmark_se_mapping_resolved.csv`; the other 971 columns are exact frequency-matrix matches.

Important caveat: these 23 candidate groups have identical frequency profiles in the available matrices, so no frequency-only method can distinguish them. The resolved labels use the strongest available semantic/glove neighborhood evidence and should be treated as the best current mapping, while preserving the evidence field for auditability.

Final resolved SE mapping: `994/994`.
