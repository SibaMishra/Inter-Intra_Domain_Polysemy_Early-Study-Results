"""Mapping check with the pipeline's own vectoriser + run on the six real requirement sets.
Usage: python3 experiments/mapping_check_and_real_sets.py [--mapping-only]
"""
import sys, os, io, re, contextlib, warnings, glob
warnings.filterwarnings('ignore')
ROOT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np, pandas as pd
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold
from ambiguity_framework import AmbiguityPipeline, _enriched_gloss_text
from domain_glossary import DOMAIN_MAPPING_BENCHMARK as B, DOMAIN_DESCRIPTORS, LEXNAME_TO_DOMAIN_PRIORS as P
from evaluate import extract_features_and_score, tune_dev_split

df = pd.read_csv(os.path.join(ROOT, 'data', 'gold_standard_250.csv'), keep_default_na=False)
df['group'] = df['target_word'].str.lower()
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
idx = list(sgkf.split(df, df['consensus_label'], df['group']))
dev = df.iloc[np.concatenate([idx[i][1] for i in range(3)])]

def mapping_eval(rep, label):
    names = rep.domain_names
    res = {'hybrid': 0, 'cosine': 0, 'prior': 0, 'always_General': 0, 'hybrid_equals_prior_top': 0}
    per_domain = {}
    for b in B:
        s = wn.synset(b['synset']); v = rep.transform([_enriched_gloss_text(s)])
        sims = cosine_similarity(v, rep.domain_vectors)[0]
        cos = {d: float(sims[i]) for i, d in enumerate(names)}
        hyb = dict(cos)
        for d, w in P.get(s.lexname(), {}).items(): hyb[d] = 0.7 * hyb[d] + 0.3 * w
        pr = P.get(s.lexname(), {})
        res['hybrid'] += max(hyb, key=hyb.get) == b['gold_domain']
        res['cosine'] += max(cos, key=cos.get) == b['gold_domain']
        res['prior'] += (max(pr, key=pr.get) if pr else 'General') == b['gold_domain']
        res['always_General'] += b['gold_domain'] == 'General'
        res['hybrid_equals_prior_top'] += bool(pr) and max(hyb, key=hyb.get) == max(pr, key=pr.get)
        d = per_domain.setdefault(b['gold_domain'], [0, 0]); d[1] += 1; d[0] += max(hyb, key=hyb.get) == b['gold_domain']
    print(f'mapping [{label}] of {len(B)}:', res, 'hybrid correct per gold domain:', per_domain)

# vectoriser of the original check
from ambiguity_framework import ContextRepresenter
r0 = ContextRepresenter(); r0.fit([b['definition'] for b in B] + list(DOMAIN_DESCRIPTORS.values()))
mapping_eval(r0, 'original check vocabulary')
pipe = AmbiguityPipeline()
with contextlib.redirect_stdout(io.StringIO()):
    pipe.fit_vocabulary(dev['requirement_sentence'].tolist())
mapping_eval(pipe.representer, 'pipeline vocabulary, primary dev split')
print('words in check:', sorted({b['word'] for b in B}))
print('lexnames with priors:', len(P), 'top prior per lexname:', {k: max(v, key=v.get) for k, v in P.items()})

if '--mapping-only' in sys.argv:
    sys.exit(0)

# ---- qualitative run on the six real sets (vocabulary: all their sentences, unlabelled) ----
with contextlib.redirect_stdout(io.StringIO()):
    dfe = extract_features_and_score(pipe, dev); sc, tau = tune_dev_split(dfe)
pipe.scorer.w_spread, pipe.scorer.w_polysemy, pipe.scorer.w_diversity = sc.w_spread, sc.w_polysemy, sc.w_diversity
pipe.scorer.threshold = tau
real = []
for f in sorted(glob.glob(os.path.join(ROOT, 'datasets', 'ds*.csv'))):
    s = pd.read_csv(f, encoding='latin-1').requirement_sentence.dropna().astype(str).tolist()
    real += [(os.path.basename(f)[:3], x) for x in s]
p2 = AmbiguityPipeline();
with contextlib.redirect_stdout(io.StringIO()):
    p2.fit_vocabulary([x for _, x in real] + dev['requirement_sentence'].tolist())
p2.scorer = pipe.scorer
rows = []
for src, sent in real:
    for r in p2.analyse_sentence(sent):
        rows.append(dict(src=src, sentence=sent, word=r.word, score=round(r.ambiguity_score, 3), marked=r.is_ambiguous,
                         type=r.ambiguity_type, senses=' || '.join(f'[{s.domain}] {s.gloss} ({s.context_similarity:.3f})' for s in r.ranked_senses[:3])))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(ROOT, 'results', 'real_sets_output.csv'), index=False)
print('real sentences', len(real), 'candidates', len(R), 'marked', int(R.marked.sum()),
      'sentences with >=1 mark', R[R.marked].sentence.nunique())
print(R[R.marked].type.value_counts().to_dict())
print(R[R.marked].word.value_counts().head(25).to_dict())
