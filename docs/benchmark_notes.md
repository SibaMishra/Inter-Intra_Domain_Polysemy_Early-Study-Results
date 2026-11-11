# Benchmark notes
* 250 items, 120 target words, six domains. Labels: Inter-Domain 110, Intra-Domain 64, None 76.
* Items were written by the authors; part of the set consists of mechanically reworded copies of base items.
* `consensus_label` is the authors' label; there is one label per item and no independent annotation.
* Grouped split: `StratifiedGroupKFold(10, shuffle=True, random_state=42)` on `target_word`; 3 folds are
  development (77 items) and 7 are test (173 items). Seeds 1 to 10 give the repeated splits.
* The TF-IDF vocabulary is fitted on development sentences only; test words absent from them get zero similarity.
