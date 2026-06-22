# Drug Mapping Audit

## Summary
- `mssf.py` exposes `--use_llm`: `True`
- `--use_llm` default is `False`: `True`
- LLM loading is gated by the flag: `True`
- `drug_names_from_FrequencyData.txt`: `759` names
- `data/drug_mapping.csv`: `757` names
- Exact overlap after normalization: `511`
- Same index on exact overlap: `1`
- Different index on exact overlap: `510`
- Potential alias/salt/form matches: `13`
- Remaining txt-only names: `235`
- Remaining mapping-only names: `233`
- `data/drug_texts.csv` rows: `757`
- Mapping -> `drug_texts.csv` name mismatches: `0`
- Mapping -> expected text mismatches: `0`
- `drug_llm_features.pt` loaded: `True`
- `drug_llm_features.pt` shape: `(757, 768)`
- LLM row count matches mapping: `True`

## Conclusion
- Risk level: `blocker_if_use_llm_enabled`
- Current LLM/text artifacts are internally consistent with drug_mapping.csv, but they are unsafe for the original MSSF drug index if drug_mapping.csv is wrong.
- This means the current text and LLM feature files follow the current `drug_mapping.csv` order, so if that mapping is wrong relative to the original 757-drug matrix, the LLM branch will attach the wrong text feature to the wrong drug index.

## Example Order Mismatches
- `Abacavir`: txt idx `710`, mapping idx `276`
- `Abiraterone`: txt idx `631`, mapping idx `258`
- `Acamprosate`: txt idx `228`, mapping idx `43`
- `Acebutolol`: txt idx `116`, mapping idx `297`
- `Aciclovir`: txt idx `449`, mapping idx `168`

## Example Potential Alias Matches
- `alendronic acid` -> `Alendronate` (`same_stemmed_core`, score `0.9800`)
- `cisatracurium` -> `Cisatracurium Besylate` (`same_core_without_salt_or_form`, score `1.0000`)
- `cyproterone` -> `Cyproterone acetate` (`same_core_without_salt_or_form`, score `1.0000`)
- `fondaparinux` -> `Fondaparinux sodium` (`same_core_without_salt_or_form`, score `1.0000`)
- `ibandronic acid` -> `Ibandronate` (`same_stemmed_core`, score `0.9800`)

## Example Remaining Txt-Only Names
- `abarelix`
- `acetylcysteine`
- `aclidinium bromide`
- `afatinib`
- `albumin`

## Example Remaining Mapping-Only Names
- `1-(3-Mercapto-2-Methyl-Propionyl)-Pyrrolidine-2-Carboxylic Acid`
- `1-(Isopropylamino)-3-(1-Naphthyloxy)-2-Propanol`
- `1-BENZYL-4-[(5,6-DIMETHOXY-1-INDANON-2-YL)METHYL]PIPERIDINE`
- `(10ALPHA,13ALPHA,14BETA,17ALPHA)-17-HYDROXYANDROST-4-EN-3-ONE`
- `(11alpha,14beta)-11,17,21-trihydroxypregn-4-ene-3,20-dione`

## Output Files
- `data/drug_mapping_audit/summary.json`
- `data/drug_mapping_audit/order_mismatches.csv`
- `data/drug_mapping_audit/potential_alias_matches.csv`
- `data/drug_mapping_audit/drug_names_only_in_frequency_txt.csv`
- `data/drug_mapping_audit/drug_names_only_in_mapping.csv`
- `data/drug_mapping_audit/index_samples.csv`
