# Inter- and Intra-Domain Polysemy in Software Requirements: Sentence-Level Prototype


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


## Third-party resource
`resources/wn-domains-3.2_mapped_to_wn30.tsv` is derived from WordNet Domains 3.2 (FBK; Bentivogli et al., 2004),
licensed under CC BY 3.0. Its labels, keyed on WordNet 2.0, were carried to WordNet 3.0 through shared sense keys from
the Collaborative Interlingual Index (Bond et al., 2016) with `experiments/build_wnd30_map.py`.
