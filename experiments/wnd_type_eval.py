"""Type evaluation with WordNet Domains as the sense-to-domain mapping.

Configurations (each refits weights and threshold on the development items):
  priors   : original hand-set lexicographer priors (reproduces Table 1)
  wnd      : WordNet Domains mapping (src/wnd_domains.py)
Type rules:
  top2     : rule (1) of the paper, domains of the two best senses
  plausible: senses within margin mu of the best context similarity (at least two);
             Inter-Domain if they span >= 2 domains, else Intra-Domain.
             Variant 'nongeneral' requires >= 2 distinct non-General domains.
             mu and the variant are tuned on development type accuracy only.
Vocabularies: development sentences ('dev') and dev + unlabelled PURE ('pure').
Usage: python3 experiments/wnd_type_eval.py
"""
import sys, os, io, re, contextlib, warnings
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from ambiguity_framework import AmbiguityPipeline, classify_ambiguity, _enriched_gloss_text
from domain_glossary import DOMAIN_DESCRIPTORS
from evaluate import prf1, extract_features_and_score, tune_dev_split
from wnd_domains import WNDDomainIdentifier

df = pd.read_csv(os.path.join(ROOT, 'data', 'gold_standard_250.csv'), keep_default_na=False)
df['group'] = df['target_word'].str.lower()
AMB = ['Inter-Domain', 'Intra-Domain']

pure = pd.read_csv(os.path.join(ROOT, 'datasets', 'Pure_Annotate_Dataset.csv'), encoding='latin-1')
pure = [s.strip() for s in pure.sentence.dropna() if len(s.split()) >= 4]
tok = lambda s: set(re.findall('[a-z]+', s.lower()))
G = [tok(s) for s in df.requirement_sentence]
pure_clean = [s for s in dict.fromkeys(pure) if not any(len(tok(s) & g) / max(1, len(tok(s) | g)) >= 0.6 for g in G)]

def gloss_corpus(pipe, sentences):
    out = set()
    for doc in pipe.preprocessor.nlp.pipe(sentences, batch_size=256):
        for c in pipe.candidate_detector.detect(doc):
            for syn in c.senses: out.add(_enriched_gloss_text(syn))
    return out
_ext = {}
def make_pipe(dev, vocab, mapping):
    p = AmbiguityPipeline()
    with contextlib.redirect_stdout(io.StringIO()):
        if vocab == 'dev':
            p.fit_vocabulary(dev['requirement_sentence'].tolist())
        else:
            if 'g' not in _ext: _ext['g'] = gloss_corpus(p, pure_clean)
            gl = gloss_corpus(p, dev['requirement_sentence'].tolist()) | _ext['g']
            p.representer.fit(dev['requirement_sentence'].tolist() + pure_clean + list(gl) + list(DOMAIN_DESCRIPTORS.values())); p._fitted = True
    if mapping == 'wnd':
        p.domain_identifier = WNDDomainIdentifier(p.domain_identifier)
    return p

def rule_plausible(rs, mu, variant):
    if not isinstance(rs, list) or len(rs) < 2: return 'Intra-Domain'
    top = rs[0].context_similarity
    P = [s for s in rs if top - s.context_similarity <= mu]
    if len(P) < 2: P = rs[:2]
    doms = {s.domain for s in P}
    if variant == 'nongeneral': doms = doms - {'General'}
    return 'Inter-Domain' if len(doms) >= 2 else 'Intra-Domain'

def types(f, pred, rule):
    out = []
    for p, (_, r) in zip(pred, f.iterrows()):
        if not p: out.append('None'); continue
        if rule[0] == 'top2':
            s = r.get('sigma_star', []); out.append(classify_ambiguity(s) if isinstance(s, list) and s else 'Intra-Domain')
        else:
            out.append(rule_plausible(r['ranked_senses'], rule[1], rule[2]))
    return out

def type_metrics(gold, pred):
    g = [x for x in gold if x != 'None']; p = [y for x, y in zip(gold, pred) if x != 'None']
    acc = np.mean([a == b for a, b in zip(g, p)])
    det = [(a, b) for a, b in zip(g, p) if b != 'None']
    acc_det = np.mean([a == b for a, b in det]) if det else np.nan
    maj_det = np.mean([a == 'Inter-Domain' for a, _ in det]) if det else np.nan
    mf1 = f1_score(g, p, labels=AMB, average='macro', zero_division=0)
    return acc, acc_det, maj_det, mf1, len(det)

RULES = [('top2',)] + [('plausible', mu, v) for mu in np.round(np.arange(0.0, 0.205, 0.02), 2) for v in ('all', 'nongeneral')]

def run(dev, test, vocab, mapping):
    p = make_pipe(dev, vocab, mapping)
    with contextlib.redirect_stdout(io.StringIO()):
        dfe = extract_features_and_score(p, dev); tfe = extract_features_and_score(p, test); sc, tau = tune_dev_split(dfe)
    S = lambda f: (sc.w_spread * f['spread'] + sc.w_polysemy * f['baseline_polysemy'] + sc.w_diversity * f['domain_diversity']).values
    pdv, pts = S(dfe) >= tau, S(tfe) >= tau
    gdev, gtest = dfe['gold_label'].values, tfe['gold_label'].values
    best = max(RULES[1:], key=lambda r: (type_metrics(gdev, types(dfe, pdv, r))[0], -r[1]))
    det = prf1(tfe['gold_is_ambig'].values, pts)
    res = dict(vocab=vocab, mapping=mapping, tau=round(tau, 2), P=det['precision'], R=det['recall'], F1=det['f1'])
    for name, rule in [('top2', RULES[0]), ('plausible', best)]:
        a, ad, md, mf, nd = type_metrics(gtest, types(tfe, pts, rule))
        res.update({f'{name}_acc': a, f'{name}_acc_detected': ad, f'{name}_macroF1': mf})
    res.update(detected_inter_share=md, n_detected=nd, plausible_rule=f'mu={best[1]},{best[2]}')
    amb = [x for x in gtest if x != 'None']
    res.update(always_inter=np.mean([x == 'Inter-Domain' for x in amb]), always_intra=np.mean([x == 'Intra-Domain' for x in amb]),
               always_inter_macroF1=f1_score(amb, ['Inter-Domain'] * len(amb), labels=AMB, average='macro', zero_division=0))
    return res

def split(seed):
    idx = list(StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed).split(df, df['consensus_label'], df['group']))
    return df.iloc[np.concatenate([idx[i][1] for i in range(3)])], df.iloc[np.concatenate([idx[i][1] for i in range(3, 10)])]

rows = []
for seed in [42] + list(range(1, 11)):
    dev, test = split(seed)
    for vocab in ('dev', 'pure'):
        for mapping in ('priors', 'wnd'):
            r = run(dev, test, vocab, mapping); r['seed'] = seed; rows.append(r)
            if seed == 42:
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    print('seed', seed, 'done', flush=True)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(ROOT, 'results', 'wnd_type_eval.csv'), index=False)
cols = ['F1', 'top2_acc', 'plausible_acc', 'top2_acc_detected', 'plausible_acc_detected', 'detected_inter_share',
        'top2_macroF1', 'plausible_macroF1', 'always_inter', 'always_inter_macroF1']
print('\nMEAN over seeds 1-10')
print(R[R.seed != 42].groupby(['vocab', 'mapping'])[cols].mean().round(3).T)
print(R[R.seed != 42].groupby(['vocab', 'mapping'])[['F1', 'top2_acc', 'plausible_acc']].std().round(3).T)
