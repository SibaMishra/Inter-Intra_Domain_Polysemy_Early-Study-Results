"""All paper numbers for the WordNet Domains configuration (main method of the paper).
Fixed grouped split (seed 42) and ten repeated splits (seeds 1-10), development vocabulary.
Prints: detection (dev F1, P, R, F1, bootstrap CI), type (rule (1) and the tied-sense variant),
w/o glossary, Spread Only typed with WordNet Domains, recall breakdown, Table 2 cases.
Usage: python3 experiments/wnd_paper_numbers.py
"""
import sys, os, io, contextlib, warnings
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from ambiguity_framework import AmbiguityPipeline, classify_ambiguity
from evaluate import prf1, extract_features_and_score, tune_dev_split, bootstrap_ci
from wnd_domains import WNDDomainIdentifier

df = pd.read_csv(os.path.join(ROOT, 'data', 'gold_standard_250.csv'), keep_default_na=False)
df['group'] = df['target_word'].str.lower()

def split(seed):
    idx = list(StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed).split(df, df['consensus_label'], df['group']))
    return df.iloc[np.concatenate([idx[i][1] for i in range(3)])], df.iloc[np.concatenate([idx[i][1] for i in range(3, 10)])]

def pipe_for(dev, use_glossary=True, mapping='wnd'):
    p = AmbiguityPipeline(use_glossary=use_glossary)
    with contextlib.redirect_stdout(io.StringIO()):
        p.fit_vocabulary(dev['requirement_sentence'].tolist())
    if mapping == 'wnd': p.domain_identifier = WNDDomainIdentifier(p.domain_identifier)
    return p

def tied(rs):
    """Rule (1) with ties: all senses sharing the best similarity vote (at least the top two)."""
    if not isinstance(rs, list) or len(rs) < 2: return 'Intra-Domain'
    P = [s for s in rs if rs[0].context_similarity - s.context_similarity <= 0]
    if len(P) < 2: P = rs[:2]
    return 'Inter-Domain' if len({s.domain for s in P}) >= 2 else 'Intra-Domain'

def types(f, pred, rule):
    out = []
    for p, (_, r) in zip(pred, f.iterrows()):
        if not p: out.append('None')
        elif rule == 'top2':
            s = r.get('sigma_star', []); out.append(classify_ambiguity(s) if isinstance(s, list) and s else 'Intra-Domain')
        else: out.append(tied(r['ranked_senses']))
    return out

def tacc(g, p):
    a = [(x, y) for x, y in zip(g, p) if x != 'None']; return np.mean([x == y for x, y in a])

def best_threshold(scores, y, grid):
    best, bf = None, -1
    for t in grid:
        f = prf1(y, scores >= t)['f1']
        if f > bf: bf, best = f, t
    return best, bf

def run(dev, test, detail=False):
    out = {}
    for name, ug in (('WND', True), ('WND w/o glossary', False)):
        p = pipe_for(dev, ug)
        with contextlib.redirect_stdout(io.StringIO()):
            dfe = extract_features_and_score(p, dev); tfe = extract_features_and_score(p, test); sc, tau = tune_dev_split(dfe)
        S = lambda f: (sc.w_spread * f['spread'] + sc.w_polysemy * f['baseline_polysemy'] + sc.w_diversity * f['domain_diversity']).values
        pd_, pt = S(dfe) >= tau, S(tfe) >= tau
        g = tfe['gold_label'].values; m = prf1(tfe['gold_is_ambig'].values, pt)
        out[name] = dict(dev_f1=prf1(dfe['gold_is_ambig'].values, pd_)['f1'], P=m['precision'], R=m['recall'], F1=m['f1'],
                         type=tacc(g, types(tfe, pt, 'top2')), type_tied=tacc(g, types(tfe, pt, 'tied')), tau=tau,
                         w=(round(sc.w_spread, 3), round(sc.w_polysemy, 3), round(sc.w_diversity, 3)))
        if name == 'WND':
            ts, tsf = best_threshold(dfe['spread'].values, dfe['gold_is_ambig'].values, np.arange(0.0, 1.0001, 0.02))
            ps = tfe['spread'].values >= ts
            out['Spread Only (WND types)'] = dict(type=tacc(g, types(tfe, ps, 'top2')), F1=prf1(tfe['gold_is_ambig'].values, ps)['f1'])
            if detail: out['_d'] = (tfe, pt, S(tfe), g, tau)
    return out

dev, test = split(42)
r = run(dev, test, detail=True); tfe, pt, Sv, g, tau = r.pop('_d')
for k, v in r.items(): print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
ci = bootstrap_ci(list(tfe['gold_is_ambig'].values), list(pt), None, None, 1000, seed=42)
print('F1 CI', [round(x, 1) for x in ci['f1_ci']])
# recall breakdown
amb = tfe[tfe.gold_label != 'None'].copy(); amb['S'] = Sv[tfe.gold_label.values != 'None']
miss = amb[amb.S < tau]
cat = lambda rr: 'not candidate' if len(rr['ranked_senses']) == 0 else ('all sim ~0' if max(x.context_similarity for x in rr['ranked_senses']) < 0.05 else 'has context')
print('missed', len(miss), miss.apply(cat, axis=1).value_counts().to_dict() if len(miss) else {})
ty = types(tfe, pt, 'top2'); det = [(a, b) for a, b in zip(g, ty) if a != 'None' and b != 'None']
print('type confusion on detected', pd.crosstab(pd.Series([a for a, _ in det], name='gold'), pd.Series([b for _, b in det], name='pred')).to_dict())
# Table 2 cases
for word, needle in [('review', 'medication review'), ('pipeline', 'trans-continental pipeline'), ('firewall', 'hardware firewall'), ('charge', 'room charge')]:
    rr = tfe[(tfe.word == word) & tfe.sentence.str.contains(needle)]
    for i, x in rr.iterrows():
        k = list(tfe.index).index(i)
        print('CASE', word, 'gold', x.gold_label, 'S', round(Sv[k], 3), 'marked', bool(pt[k]), 'type', ty[k], '|',
              ' || '.join(f'[{s.domain}] {s.gloss[:60]} ({s.context_similarity:.3f})' for s in x.ranked_senses[:3]))
        break

rows = []
for seed in range(1, 11):
    for k, v in run(*split(seed)).items():
        rows.append(dict(seed=seed, method=k, **{a: b for a, b in v.items() if a in ('F1', 'type', 'type_tied')}))
R = pd.DataFrame(rows); R.to_csv(os.path.join(ROOT, 'results', 'wnd_paper_numbers_repeats.csv'), index=False)
print(R.groupby('method')[['F1', 'type', 'type_tied']].agg(['mean', 'std']).round(3))
