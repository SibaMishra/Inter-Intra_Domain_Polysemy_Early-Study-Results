# Domain Identification & Mapping Evaluation

## 1. Executive Summary
- **Overall Accuracy**: **40.00%** (8 / 20 senses correct)
- **Evaluation Benchmark**: 20 domain-grounded WordNet synsets spanning Software/IT, Healthcare, Finance/Business, Legal/Regulatory, and General.
- **Methodology**: Evaluated using hybrid ContextRepresenter combining WordNet lexicographer files (`synset.lexname()`) with domain descriptor TF-IDF representations.

## 2. Per-Domain Performance Breakdown

```
                  precision    recall  f1-score   support

Finance/Business       0.50      0.67      0.57         3
         General       0.67      0.36      0.47        11
      Healthcare       0.33      1.00      0.50         1
Legal/Regulatory       0.00      0.00      0.00         2
      Networking       0.00      0.00      0.00         0
     Software/IT       0.20      0.33      0.25         3

        accuracy                           0.40        20
       macro avg       0.28      0.39      0.30        20
    weighted avg       0.49      0.40      0.41        20

```

## 3. Focus Words
- `port.n.01` (port, noun.location): gold General, predicted General -> correct
- `port.n.02` (port, noun.food): gold General, predicted Networking -> misclassified
- `interface.n.04` (port, noun.artifact): gold Software/IT, predicted Software/IT -> correct
- `service.n.01` (service, noun.act): gold General, predicted General -> correct
- `service.n.13` (service, noun.act): gold Legal/Regulatory, predicted Software/IT -> misclassified
- `solution.n.03` (solution, noun.cognition): gold General, predicted General -> correct
- `solution.n.01` (solution, noun.substance): gold General, predicted Healthcare -> misclassified
- `system.n.06` (system, noun.body): gold Healthcare, predicted Healthcare -> correct

## 4. Benchmark Results Table
| Synset | Word | Lexname | Gold Domain | Predicted Domain | Conf | Correct |
|---|---|---|---|---|---|---|
| `port.n.01` | port | `noun.location` | General | General | 0.21 | YES |
| `port.n.02` | port | `noun.food` | General | Networking | 0.063 | NO |
| `interface.n.04` | port | `noun.artifact` | Software/IT | Software/IT | 0.186 | YES |
| `service.n.01` | service | `noun.act` | General | General | 0.26 | YES |
| `service.n.13` | service | `noun.act` | Legal/Regulatory | Software/IT | 0.174 | NO |
| `record.n.06` | record | `noun.act` | General | Software/IT | 0.16 | NO |
| `chart.n.01` | chart | `noun.communication` | General | General | 0.16 | YES |
| `ordering.n.01` | order | `noun.group` | Software/IT | Finance/Business | 0.183 | NO |
| `order.n.07` | order | `noun.communication` | Finance/Business | Finance/Business | 0.191 | YES |
| `solution.n.03` | solution | `noun.cognition` | General | General | 0.226 | YES |
| `solution.n.01` | solution | `noun.substance` | General | Healthcare | 0.18 | NO |
| `system.n.06` | system | `noun.body` | Healthcare | Healthcare | 0.255 | YES |
| `key.n.01` | key | `noun.artifact` | General | Software/IT | 0.16 | NO |
| `table.n.02` | table | `noun.artifact` | General | Software/IT | 0.187 | NO |
| `table.n.01` | table | `noun.group` | Software/IT | General | 0.12 | NO |
| `supplier.n.01` | provider | `noun.person` | General | Healthcare | 0.176 | NO |
| `claim.n.02` | claim | `noun.communication` | General | Finance/Business | 0.143 | NO |
| `claim.n.01` | claim | `noun.communication` | Finance/Business | Finance/Business | 0.133 | YES |
| `policy.n.01` | policy | `noun.cognition` | Legal/Regulatory | General | 0.254 | NO |
| `policy.n.03` | policy | `noun.communication` | Finance/Business | Legal/Regulatory | 0.195 | NO |
