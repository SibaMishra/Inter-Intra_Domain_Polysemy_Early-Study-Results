# Inter- and Intra-Domain Polysemy in Software Requirements: Sentence-Level Prototype

Code, benchmark and results for the REFSQ 2027 Research Preview
*Distinguishing Inter- and Intra-Domain Polysemy in Software Requirements:
Early Lessons from a Sentence-Level Prototype*.

## What's Inside
| Path | Content |
|---|---|
| `src/` | `ambiguity_framework.py` (pipeline and score), `domain_glossary.py` (six domains, 40-term glossary), `evaluate.py` (metrics, tuning) |
| `data/gold_standard_250.csv` | author-written 250-item benchmark (110 Inter, 64 Intra, 76 None; 120 target words) |
| `experiments/` | one script per result of the paper |
| `results/` | CSV files and logs written by the experiments |
| `docs/` | annotation guidelines and benchmark notes |


| `datasets/` | unlabelled requirement sets: PURE sentences and six public requirement sets (used for the vocabulary and the real-requirements run) |

## Reproducing the paper
Install the dependencies (`pip install -r requirements.txt`, then `python -m spacy download en_core_web_sm`
and the NLTK `wordnet` corpus), then run from the repository root:

| Paper result | Script | Output in `results/` |
|---|---|---|
| Table 1, rows from the grouped split and the ten repeated splits | `experiments/grouped_split.py` | `grouped_split_*` |
| Table 1, PURE vocabulary row; Spread Only on the grid [0, 1] | `experiments/vocab_pure_and_repeats.py` | `vocab_pure_and_repeats_log.txt`, `repeats_dev_vs_pure.csv` |
| Why recall is low | `experiments/recall_analysis.py` | `recall_analysis_log.txt` |
| Mapping check (20 senses; 9 labelled senses not in WordNet) | `experiments/mapping_check_and_real_sets.py --mapping-only` | `mapping_check_log.txt` |
| Real-requirements run | `experiments/mapping_check_and_real_sets.py` | `mapping_check_and_real_sets_log.txt`, `real_sets_output.csv` |
| Original mapping check | `experiments/evaluate_domain_mapping.py` | `domain_mapping_*` |
| Table 2 cases | `experiments/worked_examples.py` | `worked_examples_log.txt` |
| Independent mapping (WordNet Domains): mapping check and type | `experiments/wnd_type_eval.py`, `experiments/wnd_paper_numbers.py` | `wnd_type_eval_log.txt`, `wnd_paper_numbers_log.txt` |
| 100-sentence sample for the planned annotation study | `experiments/make_annotation_sample.py` | `annotation_sample/` |

## Third-party resource
`resources/wn-domains-3.2_mapped_to_wn30.tsv` is derived from WordNet Domains 3.2 (FBK; Bentivogli et al., 2004),
licensed under CC BY 3.0. Its labels, keyed on WordNet 2.0, were carried to WordNet 3.0 through shared sense keys from
the Collaborative Interlingual Index (Bond et al., 2016) with `experiments/build_wnd30_map.py`.
