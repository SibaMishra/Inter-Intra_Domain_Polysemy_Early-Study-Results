import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src')); os.chdir(os.path.join(ROOT, 'data'))
# -*- coding: utf-8 -*-
"""
evaluate_domain_mapping.py
--------------------------
Evaluates the domain identification module (DomainIdentifier) against the
WordNet Lexicographer file benchmark and domain-annotated synsets.

Generates:
- Domain mapping accuracy, precision, recall, and macro-F1.
- Results for four focus words ("port", "service", "solution", "system").
- results/domain_mapping_evaluation.csv
- results/domain_mapping_report.md
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import wordnet as wn

from domain_glossary import DOMAIN_MAPPING_BENCHMARK, DOMAIN_DESCRIPTORS
from ambiguity_framework import ContextRepresenter, DomainIdentifier, _enriched_gloss_text


def evaluate_domain_mapping():
    print("=" * 70)
    print("EVALUATING DOMAIN IDENTIFICATION & MAPPING COMPONENT")
    print("=" * 70)

    # 1. Initialize ContextRepresenter and DomainIdentifier
    representer = ContextRepresenter()
    # Fit on all benchmark definitions + domain descriptors
    training_corpus = [b["definition"] for b in DOMAIN_MAPPING_BENCHMARK] + list(DOMAIN_DESCRIPTORS.values())
    representer.fit(training_corpus)
    domain_id = DomainIdentifier(representer)

    results = []
    focus_words = {"port", "service", "solution", "system"}
    focus_analysis = []

    for item in DOMAIN_MAPPING_BENCHMARK:
        synset_id = item["synset"]
        syn = wn.synset(synset_id)
        pred_domain, conf = domain_id.domain_of_sense(syn)
        gold_domain = item["gold_domain"]
        is_correct = (pred_domain == gold_domain)

        res = {
            "synset": synset_id,
            "word": item["word"],
            "gold_domain": gold_domain,
            "predicted_domain": pred_domain,
            "confidence": round(conf, 3),
            "lexname": syn.lexname(),
            "is_correct": is_correct,
            "definition": item["definition"]
        }
        results.append(res)

        if item["word"] in focus_words:
            focus_analysis.append(res)

    df = pd.DataFrame(results)

    # 2. Performance Metrics
    acc = accuracy_score(df["gold_domain"], df["predicted_domain"])
    report_dict = classification_report(
        df["gold_domain"], df["predicted_domain"], output_dict=True, zero_division=0
    )
    report_str = classification_report(
        df["gold_domain"], df["predicted_domain"], zero_division=0
    )

    print(f"\nOverall Domain Mapping Accuracy: {acc * 100:.2f}%")
    print("\nPer-Domain Classification Report:")
    print(report_str)

    print("-" * 70)
    print("FOCUS WORDS:")
    print("-" * 70)
    for h in focus_analysis:
        status = "CORRECT" if h["is_correct"] else "MISCLASSIFIED"
        print(f"[{status}] {h['synset']} ({h['word']}): Gold={h['gold_domain']}, Predicted={h['predicted_domain']} (lexname={h['lexname']}, conf={h['confidence']})")
        print(f"    Def: {h['definition']}")

    # 3. Save Artifacts
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    df.to_csv(os.path.join(ROOT,"results","domain_mapping_evaluation.csv"), index=False)

    focus_md = "\n".join(
        f"- `{h['synset']}` ({h['word']}, {h['lexname']}): gold {h['gold_domain']}, predicted {h['predicted_domain']}"
        f" -> {'correct' if h['is_correct'] else 'misclassified'}" for h in focus_analysis)

    md_content = f"""# Domain Identification & Mapping Evaluation

## 1. Executive Summary
- **Overall Accuracy**: **{acc * 100:.2f}%** ({df['is_correct'].sum()} / {len(df)} senses correct)
- **Evaluation Benchmark**: {len(df)} domain-grounded WordNet synsets spanning Software/IT, Healthcare, Finance/Business, Legal/Regulatory, and General.
- **Methodology**: Evaluated using hybrid ContextRepresenter combining WordNet lexicographer files (`synset.lexname()`) with domain descriptor TF-IDF representations.

## 2. Per-Domain Performance Breakdown

```
{report_str}
```

## 3. Focus Words
{focus_md}

## 4. Benchmark Results Table
| Synset | Word | Lexname | Gold Domain | Predicted Domain | Conf | Correct |
|---|---|---|---|---|---|---|
"""
    for r in results:
        md_content += f"| `{r['synset']}` | {r['word']} | `{r['lexname']}` | {r['gold_domain']} | {r['predicted_domain']} | {r['confidence']} | {'YES' if r['is_correct'] else 'NO'} |\n"

    with open(os.path.join(ROOT,'results','domain_mapping_report.md'), "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\nWrote results/domain_mapping_evaluation.csv and results/domain_mapping_report.md")
    return acc


if __name__ == "__main__":
    evaluate_domain_mapping()
