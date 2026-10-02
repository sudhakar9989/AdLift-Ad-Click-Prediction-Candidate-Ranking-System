# AdLift Feature Specification & Target Leakage Prevention

This document defines the strict feature categorization, allowlist policy, and leakage rules for predicting ad impression clicks in **AdLift**.

---

## 1. Modeling Philosophy & The Impression-Time Invariant

In real-world online advertising (DSP / SSP / Ad Exchanges), a model scores an ad candidate at the precise instant of an **ad auction / impression request**:
$$\text{Impression Opportunity} \longrightarrow \mathbf{Score}(x) \longrightarrow \text{Auction / Serve} \longrightarrow \text{User Interaction (Click)} \longrightarrow \text{Downstream Action (Conversion)}$$

### The Cardinal Rule
> **A feature is eligible for model input if and only if its value is known before or at the moment the impression is scored.**
> Any field generated during or after the user interaction is an **outcome** and constitutes catastrophic **target leakage**. Furthermore, high correlation with the target is never an excuse to include a post-impression feature.

---

## 2. Comprehensive Column Categorization

Every single one of the 22 columns in `data/raw/criteo_attribution_dataset.tsv.gz` has been categorized into one of five groups:

| Group | Meaning | Columns Assigned |
| :--- | :--- | :--- |
| **Group 1: Eligible Model Input** | Available at scoring time; safe for predictive modeling | `campaign`, `cpo`, `time_since_last_click`, `cat1`–`cat9` |
| **Group 2: Target Label** | The binary ground truth being predicted | `click` |
| **Group 3: Post-Impression / Outcome** | Downstream events occurring after the impression | `conversion`, `conversion_timestamp`, `conversion_id`, `attribution`, `click_pos`, `click_nb` |
| **Group 4: Identifier / Metadata** | Requires partitioning or specialized handling | `timestamp`, `uid` |
| **Group 5: Uncertain Timing / Meaning** | Excluded until rigorous justification is established | `cost` |

---

## 3. Detailed Column-by-Column Decisions

### Group 1: Eligible Model Inputs (Allowlisted)

1. **`campaign`** (`int64`, categorical):
   - *Status*: **APPROVED**
   - *Rationale*: At auction time, the ad server knows which advertiser campaign creative is being evaluated for placement. There are 647 distinct campaigns in the first 100k rows, making it a critical predictor of baseline creative appeal.

2. **`cpo`** (`float64`, numeric):
   - *Status*: **APPROVED**
   - *Rationale*: Cost-Per-Order (or target CPA) is a pre-configured advertiser bidding constraint entered in the campaign settings prior to the auction. It signals advertiser valuation and campaign optimization tier.

3. **`time_since_last_click`** (`int64`, numeric):
   - *Status*: **APPROVED**
   - *Rationale*: Represents the elapsed time (in seconds) since the user's *previous* recorded click before this impression occurred. Sentinel `-1` indicates no prior click was recorded. This is a pre-impression behavioral recency feature.

4. **`cat1` through `cat9`** (`int64`, categorical):
   - *Status*: **APPROVED**
   - *Rationale*: Anonymized categorical hash identifiers representing impression context (e.g., publisher domain, ad placement size, device category, geography). These arrive in the ad bid request payload and are known at inference time.

---

### Group 2: Target Label

5. **`click`** (`int64`, binary 0 or 1):
   - *Status*: **TARGET LABEL**
   - *Rationale*: The ground truth label to predict: did this impression result in a click? It must never enter model feature matrices.

---

### Group 3: Post-Impression Outcome Fields (Strictly Prohibited — Severe Leakage)

6. **`click_pos`** (`int64`):
   - *Status*: **PROHIBITED**
   - *Leakage Risk*: Extreme (100% label leakage). In the dataset, `click_pos` is `-1` whenever `click == 0`, and $\ge 0$ only when a click occurred. Using this would produce a trivial, fabricated model that simply checks if `click_pos != -1`.

7. **`click_nb`** (`int64`):
   - *Status*: **PROHIBITED**
   - *Leakage Risk*: Extreme (100% label leakage). `click_nb` is `-1` when `click == 0` and is an outcome count of session clicks.

8. **`conversion`** (`int64`):
   - *Status*: **PROHIBITED**
   - *Leakage Risk*: Severe post-click outcome. Conversions occur on the advertiser's website long after the user clicks the ad. In online serving, conversions cannot be known when scoring the ad.

9. **`conversion_timestamp`** (`int64`):
   - *Status*: **PROHIBITED**
   - *Leakage Risk*: Downstream conversion timestamp (`-1` if no conversion occurred).

10. **`conversion_id`** (`int64`):
    - *Status*: **PROHIBITED**
    - *Leakage Risk*: Downstream transaction ID generated post-conversion (`-1` if no conversion occurred).

11. **`attribution`** (`int64`):
    - *Status*: **PROHIBITED**
    - *Leakage Risk*: Outcome flag assigned by multi-touch attribution algorithms after observing all journey touchpoints.

---

### Group 4: Identifiers and Metadata (Special Handling)

12. **`timestamp`** (`int64`):
    - *Status*: **SPLIT COORDINATE ONLY (EXCLUDED FROM DIRECT MODEL INPUTS)**
    - *Rationale*: Monotonically increasing seconds from dataset origin. Passing raw linear timestamps directly to tree or linear models allows them to memorize elapsed time and overfit temporal splits rather than learning generalizable signals. Used strictly to enforce chronological Train / Validation / Test partitioning.

13. **`uid`** (`int64`):
    - *Status*: **EXCLUDED FROM BASELINE TABULAR INPUTS**
    - *Rationale*: Hashed user ID exhibits massive cardinality (86,879 unique IDs in 100k rows, $\approx 87\%$ cardinality). Direct one-hot or target encoding on cold-start users leads to severe memorization, sparsity breakdown on a laptop, and data leakage across splits.

---

### Group 5: Uncertain Timing or Meaning

14. **`cost`** (`float64`):
    - *Status*: **PROHIBITED / EXCLUDED**
    - *Rationale*: In real-time bidding exchanges (e.g. second-price or generalized first-price auctions), the actual clearing price (`cost`) is computed after all bids are collected, the auction is cleared, and the ad is served. Unless verified as an ex-ante reserve price, using realized cost introduces clearing leakage. We conservatively exclude it.

---

## 4. The Approved Feature Allowlist

The exact allowlist implemented in [`src/adlift/features.py`](file:///c:/Users/sudhakar/Desktop/anti/project%20of%20ai/SafetyTrajectory-main/src/adlift/features.py):

```python
APPROVED_NUMERIC_FEATURES = [
    "cpo",
    "time_since_last_click",
]

APPROVED_CATEGORICAL_FEATURES = [
    "campaign",
    "cat1",
    "cat2",
    "cat3",
    "cat4",
    "cat5",
    "cat6",
    "cat7",
    "cat8",
    "cat9",
]

APPROVED_INPUT_FEATURES = APPROVED_NUMERIC_FEATURES + APPROVED_CATEGORICAL_FEATURES
```

**Total Approved Input Features**: **12 features** (2 numeric: `cpo`, `time_since_last_click`; 10 categorical: `campaign` + `cat1` through `cat9`).

---

## 5. Automated Validation & Enforcement

Any model training pipeline or API scoring endpoint must call:
```python
from adlift.features import validate_feature_columns, filter_eligible_features

# Raises ValueError if any prohibited or unapproved feature is present
validate_feature_columns(input_features)

# Safely filters dataframe down to strictly approved columns
clean_df = filter_eligible_features(raw_df)
```
If any prohibited column (`click`, `conversion`, `click_pos`, `click_nb`, `cost`, etc.) enters the input stream, the validator raises an immediate `ValueError`, blocking pipeline execution.
