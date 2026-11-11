# -*- coding: utf-8 -*-
"""
wnd_domains.py
--------------
Sense-to-domain mapping from WordNet Domains 3.2 (FBK; CC BY 3.0), whose labels are keyed on
WordNet 2.0 and were carried over to WordNet 3.0 through shared sense keys
(experiments/build_wnd30_map.py, resources/wn-domains-3.2_mapped_to_wn30.tsv).

The table from WordNet Domains labels to the six domains was fixed from the label names alone,
before any evaluation. A sense with several labels takes the first one that is not General.
Senses without a label (3.5% of WordNet 3.0 nouns and verbs) are General.
"""
import os, csv

WND_TO_DOMAIN = {
    'computer_science': 'Software/IT',
    'telecommunication': 'Networking', 'telephony': 'Networking', 'telegraphy': 'Networking',
    'medicine': 'Healthcare', 'anatomy': 'Healthcare', 'physiology': 'Healthcare', 'pharmacy': 'Healthcare',
    'psychiatry': 'Healthcare', 'radiology': 'Healthcare', 'surgery': 'Healthcare', 'dentistry': 'Healthcare',
    'health': 'Healthcare',
    'law': 'Legal/Regulatory',
    'economy': 'Finance/Business', 'commerce': 'Finance/Business', 'enterprise': 'Finance/Business',
    'money': 'Finance/Business', 'finance': 'Finance/Business', 'banking': 'Finance/Business',
    'exchange': 'Finance/Business', 'insurance': 'Finance/Business', 'tax': 'Finance/Business',
    'book_keeping': 'Finance/Business',
}

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'resources', 'wn-domains-3.2_mapped_to_wn30.tsv')
_TABLE = None

def _load():
    global _TABLE
    if _TABLE is None:
        _TABLE = {r[0]: r[1].split() for r in csv.reader(open(_PATH, encoding='utf-8'), delimiter='\t')}
    return _TABLE

def wnd_labels(synset):
    return _load().get(synset.name(), [])

def wnd_domain(synset):
    for lab in wnd_labels(synset):
        d = WND_TO_DOMAIN.get(lab)
        if d: return d
    return 'General'

class WNDDomainIdentifier:
    """Drop-in replacement for ambiguity_framework.DomainIdentifier."""
    def __init__(self, base):
        self.base = base                      # keeps sentence-domain detection unchanged
    def domain_of_sentence(self, sentence):
        return self.base.domain_of_sentence(sentence)
    def domain_of_sense(self, synset):
        return wnd_domain(synset), 1.0
