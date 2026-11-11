# -*- coding: utf-8 -*-
"""
domain_glossary.py
-------------------
Domain-specific resources and lexicographer file mappings used by the framework's
"Domain Identification" module.

- Uses WordNet lexicographer files (synset.lexname()) as hand-set domain priors.
- Combines these priors with TF-IDF similarity to short domain descriptors.
- Holds the 40-term candidate glossary and the 29-sense domain-mapping check.
"""

from typing import Dict, List, Tuple, Optional
from nltk.corpus import wordnet as wn

# --------------------------------------------------------------------------
# 1. Target Application Domains & Representative Vocabulary Descriptors
# --------------------------------------------------------------------------
DOMAIN_DESCRIPTORS = {
    "Software/IT": (
        "system software application database server client interface "
        "program code module function process file record field key "
        "table index session login password network protocol port "
        "screen user interface data storage memory cache log error "
        "exception configuration parameter query input output display "
        "browser website application programming interface api architecture "
        "endpoint socket component algorithm service thread deployment"
    ),
    "Healthcare": (
        "patient physician nurse clinician doctor diagnosis medication "
        "prescription treatment allergy immunization vaccine provider "
        "clinic hospital chart record encounter visit vital sign lab "
        "laboratory result problem list care plan discharge admission "
        "medical history symptom disease condition dosage drug therapy "
        "clinical biometric pathology triage electronic health ehr emr"
    ),
    "Networking": (
        "network port protocol packet router switch firewall bandwidth "
        "connection socket ethernet ip address gateway node latency "
        "transmission wireless server client host traffic subnet broadcast "
        "routing interface packet tcp udp wifi communication channel"
    ),
    "Legal/Regulatory": (
        "policy regulation compliance law statute rule authorization "
        "consent privacy confidentiality hipaa audit disclosure "
        "requirement standard certification obligation liability "
        "governance security rule breach notification penalty contractual legal"
    ),
    "Finance/Business": (
        "payment billing invoice account balance transaction cost fee "
        "insurance claim reimbursement charge revenue budget contract "
        "vendor procurement price rate financial statement fiscal ledger "
        "order purchase asset liability currency payroll audit"
    ),
    "General": (
        "information process activity item entity object event action "
        "operation task step procedure method report document concept "
        "structure physical location vessel person group unit relation"
    ),
}

# --------------------------------------------------------------------------
# 2. WordNet Lexicographer Category (lexname) Priors
# --------------------------------------------------------------------------
# WordNet partitions synsets into 45 lexicographer files. We map these
# semantic categories to baseline application domain likelihoods:
LEXNAME_TO_DOMAIN_PRIORS: Dict[str, Dict[str, float]] = {
    "noun.body": {"Healthcare": 0.85, "General": 0.15},
    "noun.state": {"Healthcare": 0.50, "General": 0.35, "Software/IT": 0.15},
    "noun.substance": {"Healthcare": 0.60, "General": 0.40},
    "noun.artifact": {"Software/IT": 0.45, "Networking": 0.25, "General": 0.30},
    "noun.communication": {"Software/IT": 0.40, "Legal/Regulatory": 0.25, "General": 0.35},
    "noun.possession": {"Finance/Business": 0.80, "Legal/Regulatory": 0.10, "General": 0.10},
    "noun.act": {"Software/IT": 0.35, "Healthcare": 0.25, "General": 0.40},
    "noun.cognition": {"Software/IT": 0.35, "General": 0.65},
    "noun.group": {"Finance/Business": 0.35, "Legal/Regulatory": 0.25, "General": 0.40},
    "noun.location": {"General": 0.70, "Networking": 0.20, "Software/IT": 0.10},
    "noun.time": {"General": 0.80, "Software/IT": 0.20},
    "noun.person": {"Healthcare": 0.45, "General": 0.55},
    "noun.attribute": {"Software/IT": 0.35, "Legal/Regulatory": 0.25, "General": 0.40},
}

# --------------------------------------------------------------------------
# 3. Curated Candidate Polysemous Terms
# --------------------------------------------------------------------------
# Known cross-domain terms from RE literature & PURE corpus inspection
CANDIDATE_GLOSSARY_TERMS = {
    "record", "chart", "note", "session", "key", "field", "plan", "order",
    "code", "history", "discharge", "provider", "claim", "policy", "port",
    "table", "log", "exchange", "transaction", "profile",
    "summary", "report", "flag", "status", "role", "account", "identifier",
    "authorization", "reference", "template", "form", "review", "alert",
    "message", "queue", "encounter", "solution", "service", "system", "client"
}

# --------------------------------------------------------------------------
# 4. Domain Mapping Check
# --------------------------------------------------------------------------
# Senses of 12 benchmark words, each with the domain the authors assigned.
# "synset" is the WordNet 3.0 sense whose definition expresses the labelled meaning;
# "definition" is that WordNet definition and "labelled_meaning" the authors' wording.
# An earlier version paired 25 of 29 labels with the wrong synset IDs; they were
# matched by hand to the correct senses. Nine labelled meanings have no WordNet
# sense at all and are listed in MAPPING_CHECK_NOT_IN_WORDNET instead.
DOMAIN_MAPPING_BENCHMARK = [
    {"synset": "port.n.01", "word": "port", "gold_domain": "General", "definition": "a place (seaport or airport) where people and merchandise can enter or leave a country", "labelled_meaning": "a place on a waterway with facilities for loading and unloading ships"},
    {"synset": "port.n.02", "word": "port", "gold_domain": "General", "definition": "sweet dark-red dessert wine originally from Portugal", "labelled_meaning": "sweet dark-red dessert wine"},
    {"synset": "interface.n.04", "word": "port", "gold_domain": "Software/IT", "definition": "(computer science) computer circuit consisting of the hardware and associated circuitry that links one device with another (especially a computer and a hard disk drive or other peripherals)", "labelled_meaning": "an interface on a computer to which a device can be connected"},
    {"synset": "service.n.01", "word": "service", "gold_domain": "General", "definition": "work done by one person or group that benefits another", "labelled_meaning": "work done by one person or group that benefits another"},
    {"synset": "service.n.13", "word": "service", "gold_domain": "Legal/Regulatory", "definition": "the act of delivering a writ or summons upon someone", "labelled_meaning": "the act of delivering a formal notice or writ"},
    {"synset": "record.n.06", "word": "record", "gold_domain": "General", "definition": "an extreme attainment; the best (or worst) performance ever attested (as in a sport)", "labelled_meaning": "the extreme ranking attained in sports or achievements"},
    {"synset": "chart.n.01", "word": "chart", "gold_domain": "General", "definition": "a visual display of information", "labelled_meaning": "a visual display of information like a graph or table"},
    {"synset": "ordering.n.01", "word": "order", "gold_domain": "Software/IT", "definition": "logical or comprehensible arrangement of separate elements", "labelled_meaning": "logical arrangement or sequence of items"},
    {"synset": "order.n.07", "word": "order", "gold_domain": "Finance/Business", "definition": "a commercial document used to request someone to supply something in return for payment and providing specifications and quantities", "labelled_meaning": "a commercial document requesting supply of goods or services"},
    {"synset": "solution.n.03", "word": "solution", "gold_domain": "General", "definition": "a method for solving a problem", "labelled_meaning": "a method or process of solving a problem"},
    {"synset": "solution.n.01", "word": "solution", "gold_domain": "General", "definition": "a homogeneous mixture of two or more substances; frequently (but not necessarily) a liquid solution", "labelled_meaning": "a homogeneous mixture of two or more substances in chemistry"},
    {"synset": "system.n.06", "word": "system", "gold_domain": "Healthcare", "definition": "a group of physiologically or anatomically related organs or parts", "labelled_meaning": "a group of physiologically or anatomically related organs"},
    {"synset": "key.n.01", "word": "key", "gold_domain": "General", "definition": "metal device shaped in such a way that when it is inserted into the appropriate lock the lock's mechanism can be rotated", "labelled_meaning": "metal implement used to open a physical lock"},
    {"synset": "table.n.02", "word": "table", "gold_domain": "General", "definition": "a piece of furniture having a smooth flat top that is usually supported by one or more vertical legs", "labelled_meaning": "a piece of furniture having a smooth flat top supported by legs"},
    {"synset": "table.n.01", "word": "table", "gold_domain": "Software/IT", "definition": "a set of data arranged in rows and columns", "labelled_meaning": "an organized set of data elements arranged in rows and columns"},
    {"synset": "supplier.n.01", "word": "provider", "gold_domain": "General", "definition": "someone whose business is to supply a particular service or commodity", "labelled_meaning": "someone who provides or supplies a resource"},
    {"synset": "claim.n.02", "word": "claim", "gold_domain": "General", "definition": "an assertion that something is true or factual", "labelled_meaning": "an assertion that something is true or a factual statement"},
    {"synset": "claim.n.01", "word": "claim", "gold_domain": "Finance/Business", "definition": "an assertion of a right (as to money or property)", "labelled_meaning": "a formal request to an insurance company for payment or reimbursement"},
    {"synset": "policy.n.01", "word": "policy", "gold_domain": "Legal/Regulatory", "definition": "a plan of action adopted by an individual or social group", "labelled_meaning": "a plan of action adopted by a government or organization"},
    {"synset": "policy.n.03", "word": "policy", "gold_domain": "Finance/Business", "definition": "written contract or certificate of insurance", "labelled_meaning": "written insurance contract outlining terms of coverage"},
]

# Labelled meanings with no WordNet 3.0 sense (6 Software/IT, 3 Healthcare).
MAPPING_CHECK_NOT_IN_WORDNET = [
    {"word": "service", "gold_domain": "Software/IT", "labelled_meaning": "a service program operating under an operating system"},
    {"word": "record", "gold_domain": "Software/IT", "labelled_meaning": "a document or data structure containing factual information"},
    {"word": "chart", "gold_domain": "Healthcare", "labelled_meaning": "a medical record documenting a patient's clinical history and care"},
    {"word": "order", "gold_domain": "Healthcare", "labelled_meaning": "a physician's formal instruction for a medication or diagnostic test"},
    {"word": "solution", "gold_domain": "Software/IT", "labelled_meaning": "a packaged software application or system solving business needs"},
    {"word": "system", "gold_domain": "Software/IT", "labelled_meaning": "a group of independent computer hardware and software components"},
    {"word": "key", "gold_domain": "Software/IT", "labelled_meaning": "a database column or code that serves as an identifier or decryptor"},
    {"word": "provider", "gold_domain": "Healthcare", "labelled_meaning": "a licensed healthcare professional or medical institution delivering care"},
    {"word": "policy", "gold_domain": "Software/IT", "labelled_meaning": "a programmatic rule governing security access and authorization permissions"},
]
