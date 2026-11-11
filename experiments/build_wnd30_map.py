"""Map WordNet Domains 3.2 labels (keyed on WordNet 2.0 offsets) to WordNet 3.0 synsets.
Route: WN2.0 offset -> sense keys (cili/sense-mappings/pwn20.tab) -> WN3.0 synset via NLTK
(wn.lemma_from_key). Each 3.0 synset takes the domains of the 2.0 synset that supplies most
of its sense keys. Nouns and verbs only.
Usage: python3 experiments/build_wnd30_map.py <wn-domains-3.2-20070223> <cili/sense-mappings/pwn20.tab> <out.tsv>
"""
import sys, collections
from nltk.corpus import wordnet as wn
wnd_file, map20, out = sys.argv[1:4]

dom20 = {}
for line in open(wnd_file, encoding='latin-1'):
    off, doms = line.rstrip('\n').split('\t')
    dom20[off] = doms.split()

keys20 = collections.defaultdict(list)
for line in open(map20, encoding='utf-8', errors='ignore'):
    off, key = line.rstrip('\n').split('\t')
    if off[-1] in 'nv':
        keys20[off].append(key)

votes = collections.defaultdict(collections.Counter)   # wn30 synset name -> Counter(wn20 offset)
miss = 0
for off, keys in keys20.items():
    if off not in dom20: continue
    for k in keys:
        try:
            syn = wn.lemma_from_key(k).synset()
        except Exception:
            try:
                syn = wn.lemma_from_key(k.lower()).synset()
            except Exception:
                miss += 1; continue
        votes[syn.name()][off] += 1

with open(out, 'w') as f:
    for name, c in votes.items():
        off, _ = c.most_common(1)[0]
        f.write(f"{name}\t{' '.join(dom20[off])}\t{off}\n")
total30 = sum(1 for p in 'nv' for _ in wn.all_synsets(p))
print('wn20 synsets with domains:', len(dom20), '| wn30 noun+verb synsets mapped:', len(votes), 'of', total30,
      f'({len(votes) / total30:.1%}) | unresolved keys:', miss)
