"""New experiments for the REFSQ revision.
Usage: python3 experiments/vocab_pure_and_repeats.py
1. Reproduce primary grouped split (dev-only vocabulary).
2. Same split + seeds 1-10 with vocabulary fitted on PURE (unlabelled, near-duplicates removed) + dev.
3. Recall-loss breakdown under both vocabularies.
4. Domain-mapping check with the pipeline's own vectoriser, and with prior-only / cosine-only mapping.
"""
import sys, os, io, re, contextlib, warnings, json
warnings.filterwarnings('ignore')
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from nltk.corpus import wordnet as wn
import ambiguity_framework as af
from ambiguity_framework import AmbiguityPipeline, classify_ambiguity, _enriched_gloss_text
from domain_glossary import DOMAIN_DESCRIPTORS, DOMAIN_MAPPING_BENCHMARK, LEXNAME_TO_DOMAIN_PRIORS
from evaluate import prf1, extract_features_and_score, tune_dev_split, bootstrap_ci

df = pd.read_csv(os.path.join(ROOT, 'data', 'gold_standard_250.csv'), keep_default_na=False)
df['group'] = df['target_word'].str.lower()

# ---- external unlabelled corpus: PURE, minus near-duplicates of benchmark items ----
pure = pd.read_csv(os.path.join(ROOT, 'datasets', 'Pure_Annotate_Dataset.csv'), encoding='latin-1')
pure = [s.strip() for s in pure.sentence.dropna() if len(s.split()) >= 4]
tok = lambda s: set(re.findall('[a-z]+', s.lower()))
G = [tok(s) for s in df.requirement_sentence]
def near(s):
    t = tok(s)
    return any(len(t & g) / max(1, len(t | g)) >= 0.6 for g in G)
pure_clean = [s for s in dict.fromkeys(pure) if not near(s)]
print('PURE sentences', len(pure), 'unique+clean', len(pure_clean))

def gloss_corpus(pipe, sentences):
    out = set()
    for doc in pipe.preprocessor.nlp.pipe(sentences, batch_size=256) if hasattr(pipe.preprocessor, 'nlp') else map(pipe.preprocessor.process, sentences):
        for c in pipe.candidate_detector.detect(doc):
            for syn in c.senses: out.add(_enriched_gloss_text(syn))
    return out

_cache = {}
def ext_glosses(use_glossary):
    if use_glossary not in _cache:
        p = AmbiguityPipeline(use_glossary=use_glossary)
        _cache[use_glossary] = gloss_corpus(p, pure_clean)
    return _cache[use_glossary]

def make_pipe(dev, vocab, use_glossary=True):
    p = AmbiguityPipeline(use_glossary=use_glossary)
    with contextlib.redirect_stdout(io.StringIO()):
        if vocab == 'dev':
            p.fit_vocabulary(dev['requirement_sentence'].tolist())
        else:
            sents = dev['requirement_sentence'].tolist() + pure_clean
            gl = gloss_corpus(p, dev['requirement_sentence'].tolist()) | ext_glosses(use_glossary)
            p.representer.fit(sents + list(gl) + list(DOMAIN_DESCRIPTORS.values())); p._fitted = True
    return p

def types_from(f, pred):
    out = []
    for p, (_, r) in zip(pred, f.iterrows()):
        if not p: out.append('None')
        else:
            s = r.get('sigma_star', [])
            out.append(classify_ambiguity(s) if isinstance(s, list) and s else 'Intra-Domain')
    return out
def type_acc(gold, pred):
    amb = [(g, p) for g, p in zip(gold, pred) if g != 'None']
    return sum(g == p for g, p in amb) / len(amb)
def best_threshold(scores, y, grid):
    best, bf = None, -1
    for t in grid:
        f = prf1(y, scores >= t)['f1']
        if f > bf: bf, best = f, t
    return best, bf

def run(dev, test, vocab, detail=False):
    ydev = dev['consensus_label'].ne('None').values; ytest = test['consensus_label'].ne('None').values
    gold = test['consensus_label'].values; res = {}
    for name, ug in [('Proposed', True), ('No glossary', False)]:
        p = make_pipe(dev, vocab, ug)
        with contextlib.redirect_stdout(io.StringIO()):
            dfe = extract_features_and_score(p, dev); tfe = extract_features_and_score(p, test)
            sc, tau = tune_dev_split(dfe)
        S = lambda f: (sc.w_spread*f['spread'] + sc.w_polysemy*f['baseline_polysemy'] + sc.w_diversity*f['domain_diversity']).values
        pred = S(tfe) >= tau; ty = types_from(tfe, pred)
        res[name] = dict(dev_f1=prf1(ydev, S(dfe) >= tau)['f1'], **prf1(ytest, pred), type=type_acc(gold, ty),
                         param=f'tau={tau:.2f} w=({sc.w_spread:.3f},{sc.w_polysemy:.3f},{sc.w_diversity:.3f})')
        if name == 'Proposed':
            ts, tsf = best_threshold(dfe['spread'].values, ydev, np.arange(0.0, 1.0001, 0.02))
            ps = tfe['spread'].values >= ts
            res['Spread only'] = dict(dev_f1=tsf, **prf1(ytest, ps), type=type_acc(gold, types_from(tfe, ps)), param=f'tau={ts:.2f}')
            if detail:
                res['_detail'] = dict(tfe=tfe, pred=pred, types=ty, gold=gold, ytest=ytest, tau=tau, S=S(tfe))
    amb = [g for g in gold if g != 'None']
    res['_always_inter'] = amb.count('Inter-Domain') / len(amb); res['_always_intra'] = amb.count('Intra-Domain') / len(amb)
    return res

def split(seed, folds=10, dev_folds=3):
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    idx = list(sgkf.split(df, df['consensus_label'], df['group']))
    dev_idx = np.concatenate([idx[i][1] for i in range(dev_folds)]); test_idx = np.concatenate([idx[i][1] for i in range(dev_folds, folds)])
    return df.iloc[dev_idx], df.iloc[test_idx]

def recall_breakdown(d):
    tfe, pred = d['tfe'], d['pred']
    c = dict(not_candidate=0, zero_context=0, below_tau=0, zero_context_marked=0)
    for (_, r), p in zip(tfe.iterrows(), pred):
        if not r['gold_is_ambig']: continue
        rs = r['ranked_senses'] if isinstance(r['ranked_senses'], list) else []
        zero = bool(rs) and max(s.context_similarity for s in rs) < 0.04
        if p:
            c['zero_context_marked'] += zero; continue
        if not rs: c['not_candidate'] += 1
        elif zero: c['zero_context'] += 1
        else: c['below_tau'] += 1
    return c

out = {}
dev, test = split(42)
print('primary dev', len(dev), 'test', len(test))
for vocab in ['dev', 'pure']:
    r = run(dev, test, vocab, detail=True); d = r.pop('_detail')
    print(f'\n== primary split, vocabulary={vocab}  always-Inter {r["_always_inter"]:.3f} always-Intra {r["_always_intra"]:.3f}')
    for k, v in r.items():
        if k.startswith('_'): continue
        print(f"{k:12s} devF1 {v['dev_f1']:.3f} P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f} type {v['type']:.3f} {v['param']}")
    ci = bootstrap_ci(list(d['ytest']), list(d['pred']), list(d['gold']), list(d['types']), 1000, seed=42)
    print('CI', {k: (round(a*100, 1), round(b*100, 1)) for k, (a, b) in ci.items()})
    m = prf1(d['ytest'], d['pred'])
    nc = sum(1 for g, p in zip(d['gold'], d['types']) if g != 'None' and g == p)
    print('tp fp fn tn', m['tp'], m['fp'], m['fn'], m['tn'], 'typed correct', nc, 'recall breakdown', recall_breakdown(d))
    # type confusion among detected
    conf = pd.crosstab(pd.Series([g for g, p in zip(d['gold'], d['types']) if g != 'None' and p != 'None'], name='gold'),
                       pd.Series([p for g, p in zip(d['gold'], d['types']) if g != 'None' and p != 'None'], name='pred'))
    print(conf)
    out[f'primary_{vocab}'] = {k: v for k, v in r.items()}
    if vocab == 'dev': primary_pipe_dev = d

rows = []
for seed in range(1, 11):
    dv, ts_ = split(seed)
    for vocab in ['dev', 'pure']:
        r = run(dv, ts_, vocab)
        for k, v in r.items():
            if k.startswith('_'): continue
            rows.append(dict(seed=seed, vocab=vocab, method=k, f1=v['f1'], prec=v['precision'], rec=v['recall'], type=v['type']))
        rows.append(dict(seed=seed, vocab=vocab, method='always-Intra', type=r['_always_intra']))
        rows.append(dict(seed=seed, vocab=vocab, method='always-Inter', type=r['_always_inter']))
    print('seed', seed, 'done', flush=True)
a = pd.DataFrame(rows)
a.to_csv(os.path.join(ROOT, 'results', 'repeats_dev_vs_pure.csv'), index=False)
print(a.groupby(['vocab', 'method'])[['f1', 'prec', 'rec', 'type']].agg(['mean', 'std']).round(3))
