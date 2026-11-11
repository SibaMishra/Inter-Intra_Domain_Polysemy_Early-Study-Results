"""Sample 100 real requirement sentences (stratified by set, seed 2027) and record every
candidate word the prototype considers, with its mark, type, score and leading senses.
Same setup as the paper's real-requirements run: weights and threshold from the primary
grouped split, vocabulary fitted on the six real sets plus the development sentences.
Usage: python3 experiments/make_annotation_sample.py [out_dir]
"""
import sys, os, io, glob, contextlib, warnings
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results', 'annotation_sample')
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from ambiguity_framework import AmbiguityPipeline
from evaluate import extract_features_and_score, tune_dev_split

df = pd.read_csv(os.path.join(ROOT, 'data', 'gold_standard_250.csv'), keep_default_na=False)
df['group'] = df['target_word'].str.lower()
idx = list(StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42).split(df, df['consensus_label'], df['group']))
dev = df.iloc[np.concatenate([idx[i][1] for i in range(3)])]

real = []
for f in sorted(glob.glob(os.path.join(ROOT, 'datasets', 'ds*.csv'))):
    name = os.path.basename(f).replace('.csv', '')
    for x in pd.read_csv(f, encoding='latin-1').requirement_sentence.dropna().astype(str):
        real.append((name, ' '.join(x.split())))
R = pd.DataFrame(real, columns=['set', 'sentence']).drop_duplicates('sentence').reset_index(drop=True)

# stratified sample of 100 sentences, proportional to set size
rng = np.random.default_rng(2027)
quota = (R.set.value_counts() / len(R) * 100).round().astype(int)
while quota.sum() != 100:
    quota[quota.idxmax() if quota.sum() > 100 else quota.idxmin()] += -1 if quota.sum() > 100 else 1
picked = []
for s, q in quota.items():
    rows = R.index[R.set == s].to_numpy()
    picked += list(rng.choice(rows, size=q, replace=False))
S = R.loc[sorted(picked)].reset_index(drop=True)
S.insert(0, 'sid', [f'S{i+1:03d}' for i in range(len(S))])
print('quota', quota.to_dict(), 'sampled', len(S))

with contextlib.redirect_stdout(io.StringIO()):
    p0 = AmbiguityPipeline(); p0.fit_vocabulary(dev['requirement_sentence'].tolist())
    sc, tau = tune_dev_split(extract_features_and_score(p0, dev))
    pipe = AmbiguityPipeline(); pipe.fit_vocabulary(R.sentence.tolist() + dev['requirement_sentence'].tolist())
pipe.scorer.w_spread, pipe.scorer.w_polysemy, pipe.scorer.w_diversity, pipe.scorer.threshold = sc.w_spread, sc.w_polysemy, sc.w_diversity, tau
print('tau', round(tau, 2), 'weights', round(sc.w_spread, 3), round(sc.w_polysemy, 3), round(sc.w_diversity, 3))

rows = []
for _, r in S.iterrows():
    doc = pipe.preprocessor.process(r.sentence)
    cands = pipe.candidate_detector.detect(doc)
    res = pipe.analyse_sentence(r.sentence)
    for k, (c, a) in enumerate(zip(cands, res)):
        assert c.lemma == a.word
        tok = doc[c.token_index]
        rows.append(dict(sid=r.sid, wid=f'{r.sid}-{k+1:02d}', set=r.set, sentence=r.sentence,
                         word=tok.text, lemma=c.lemma, char_start=tok.idx,
                         sys_marked=bool(a.is_ambiguous), sys_type=a.ambiguity_type, sys_score=round(a.ambiguity_score, 3),
                         sys_senses=' || '.join(f'[{s.domain}] {s.gloss}' for s in a.ranked_senses[:3])))
W = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
S.to_csv(os.path.join(OUT, 'sample_sentences.csv'), index=False)
W.to_csv(os.path.join(OUT, 'system_key.csv'), index=False)
print('candidate words', len(W), 'marked', int(W.sys_marked.sum()),
      'marks per sentence', round(W.sys_marked.sum() / len(S), 2),
      'sentences with a mark', W[W.sys_marked].sid.nunique())
print(W[W.sys_marked].sys_type.value_counts().to_dict())
