# NPV (Negative Predictive Value) Explained

## What is NPV?

**NPV answers the question**: *"If my model predicts a patient is MSS (negative), what's the probability they are truly MSS?"*

## The Formula

```
NPV = TN / (TN + FN)
```

Where:
- **TN (True Negatives)** = Patients predicted MSS who are truly MSS ✓ CORRECT!
- **FN (False Negatives)** = Patients predicted MSS who are actually MSI-H ✗ MISSED!

## Visual Example

Let's say you have 100 patients your model predicts as MSS:

```
┌─────────────────────────────────────────────────────────┐
│         100 Patients Predicted MSS (Negative)           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓  │
│  ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓ ✗✗✗✗✗✗✗✗✗✗✗✗✗  │
│                                                         │
│  ✓ = 87 patients truly MSS (TN = 87)                   │
│  ✗ = 13 patients actually MSI-H (FN = 13) - DANGEROUS! │
│                                                         │
│  NPV = 87 / (87 + 13) = 87 / 100 = 0.87 = 87%         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Interpretation**: If NPV = 87%, then **13% of your "MSS" calls are wrong** - these are MSI-H patients who won't get the treatment they need!

## Why NPV Matters for Rule-Out Tests

For MSI testing used as a **rule-out test** (to confidently say "this patient is MSS"):
- High NPV is **critical** because missed MSI-H cases (false negatives) are dangerous
- These patients won't get immunotherapy they might benefit from
- **Target: NPV > 99%** means less than 1 in 100 "MSS" calls are wrong

## The Confusion Matrix Connection

```
                    Predicted
                MSS         MSI-H
              ┌─────────┬─────────┐
Actual  MSS   │   TN    │   FP    │  Specificity = TN/(TN+FP)
              ├─────────┼─────────┤
      MSI-H   │   FN    │   TP    │  Sensitivity = TP/(TP+FN)
              └─────────┴─────────┘
                 │         │
                NPV       PPV
```

**NPV = TN / (TN + FN)** - Bottom row, left column
**PPV = TP / (TP + FP)** - Right column, top row

## Real AORTIC Results Example

### HistoBistro Model (Threshold = 0.5)

```
Confusion Matrix:
                Predicted
                MSS     MSI-H
              ┌─────┬───────┐
Actual  MSS   │ 147 │  57   │  204 total MSS
              ├─────┼───────┤
      MSI-H   │  23 │  36   │   59 total MSI-H
              └─────┴───────┘
                170    93
```

**NPV Calculation**:
- TN = 147 (correctly identified MSS)
- FN = 23 (missed MSI-H cases - called them MSS!)
- Total predicted MSS = 147 + 23 = 170

```
NPV = 147 / (147 + 23)
    = 147 / 170
    = 0.865
    = 86.5%
```

**Clinical Interpretation**:
- Out of 170 patients called MSS, 23 are actually MSI-H
- **Miss rate = 23/170 = 13.5%**
- This means 13-14 out of every 100 patients called MSS actually have MSI-H
- **This is far from the target NPV > 99%!**

## NPV vs Sensitivity Trade-off

There's a fundamental trade-off:
- **High Sensitivity** (catch all MSI-H) → more patients called MSI-H → fewer predicted MSS → NPV can be high but you're calling most people positive
- **High Specificity** (confident in MSS calls) → more patients called MSS → higher chance of missing some MSI-H → lower NPV

You can adjust this by changing the decision threshold:
- **Lower threshold** (e.g., 0.3): Predict MSI-H more liberally → higher sensitivity, lower NPV
- **Higher threshold** (e.g., 0.7): Predict MSS more liberally → higher specificity, but if you miss any MSI-H, NPV drops

## Current AORTIC Results (Threshold = 0.5)

| Model | NPV | Missed MSI-H Rate | Interpretation |
|-------|-----|-------------------|----------------|
| **CNN** | 85.9% | 14.1% | Out of 100 patients called MSS, 14 are actually MSI-H |
| **HistoBistro** | 86.5% | 13.5% | Out of 100 patients called MSS, 13-14 are actually MSI-H |
| **CTransPath** | 85.5% | 14.5% | Out of 100 patients called MSS, 14-15 are actually MSI-H |
| **H-Optimus-0** | 77.2% | 22.8% | Out of 100 patients called MSS, 23 are actually MSI-H |

## The Goal

For a clinically useful rule-out test:
- **Target: NPV > 99%**
- This means: Less than 1 in 100 "MSS" calls are wrong
- Or: Miss rate < 1%

**None of the current models achieve this target**, suggesting:
1. Population-specific training alone is not sufficient
2. May need different thresholds, additional features, or algorithmic improvements
3. The pre-trained Western model (HistoBistro) performs similarly to Nigerian-trained models

## Summary

- **NPV = Confidence in negative (MSS) predictions**
- **High NPV** = Few missed MSI-H cases = Safe to rule out MSI-H
- **Low NPV** = Many missed MSI-H cases = NOT safe as a rule-out test
- **Current models** achieve ~85-87% NPV, missing 13-15% of MSI-H cases
- **This is clinically insufficient** for a rule-out test (target: >99%)
