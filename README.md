# Job Posting Fraud Detection

NLP pipeline on 17,880 job postings (4.8% fraudulent) that compares TF-IDF baselines (Logistic Regression, Random Forest, LinearSVC) with a fine-tuned DistilBERT, and adds SHAP/LIME explainability for the baseline, an ablation study, robustness checks and an interactive Gradio demo.

## Business Problem

Online job platforms receive large numbers of postings, and fraudulent listings, designed to collect personal information, extract money or deceive job seekers, are hard to catch by manual review. This project builds a classifier that labels postings as real or fraudulent so that suspicious ones can be flagged for review.

Using illustrative error costs (100 per missed fake posting, 10 per real posting wrongly flagged), the Logistic Regression baseline reduces total cost by **84.2%** compared with a no-model baseline, and DistilBERT by **85.8%**. Scaled to 1 million postings with the same fraud rate and error rates, the baseline would intercept about 41,600 fake listings but would also flag about 12,300 genuine ones for review. These figures depend entirely on the assumed costs.

## Dataset

- **Source:** [Real / Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) (Kaggle)
- **Size:** 17,880 job postings
- **Target:** binary, Real (95.2%) vs Fake (4.8%)
- **Challenge:** severe class imbalance; the test set contains only 173 fake postings

## Approach

| Step | Description |
|------|-------------|
| EDA | Class imbalance, text length distributions, word clouds, categorical patterns (employment type, experience level, company logo) |
| Preprocessing | HTML cleaning, lowercasing, removal of punctuation, digits and stopwords; one text field combining title, company profile, description and requirements |
| Feature Engineering | 13 structural and text-derived features (e.g. has_company_logo, has_company_profile, word_count) combined with TF-IDF |
| Baseline Models | TF-IDF (n-grams 1-3) with Logistic Regression, Random Forest and calibrated LinearSVC |
| Imbalanced Data Handling | Class weighting, SMOTE and threshold tuning compared on Logistic Regression (threshold chosen on out-of-fold training predictions only) |
| Advanced Model | DistilBERT fine-tuning with class-weighted loss and early stopping on a validation set carved from the training data |
| Explainability | SHAP (global) and LIME (single posting) for the Logistic Regression baseline |
| Advanced NLP | Exploratory analysis of recurring phrases, sentiment, readability and named entities |
| Ablation Study | Logistic Regression trained on one input type at a time (text, metadata, title, description) vs all combined |
| Robustness | Stratified 5-fold cross-validation, calibration curve, bootstrap 95% confidence interval |
| Error Analysis | Profile of the false negatives and false positives of the Logistic Regression baseline |
| Business Framing | Cost-benefit analysis with illustrative asymmetric error costs and a simulation scaled to 1M postings |
| Demo | Interactive Gradio demo of the Logistic Regression baseline (temporary public link) |

## Results

Fake class, evaluated on a held-out test set of 3,576 postings (173 fake):

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| Logistic Regression | 0.77 | 0.87 | 0.82 | 0.991 |
| Random Forest | 1.00 | 0.58 | 0.74 | 0.990 |
| LinearSVC | 0.97 | 0.76 | 0.85 | 0.991 |
| **DistilBERT** | **0.91** | **0.87** | **0.89** | **0.990** |

DistilBERT has the highest F1. It catches the same number of fake postings as the Logistic Regression baseline (150 of 173) with far fewer false alarms (15 false positives vs 44), so its gain comes from precision. Its margin over LinearSVC (F1 0.89 vs 0.85) is small relative to the uncertainty of this test set (about ±0.04 in F1).

## Key Findings

- Real postings include a company logo in 81.9% of cases vs 32.7% for fake ones, and a company profile in 84.0% vs 32.2%.
- In the Logistic Regression baseline, the five most influential features according to SHAP are structural: company logo, company profile, screening questions, required education and salary range.
- Text carries most of the predictive signal: text alone reaches F1 0.833 vs 0.817 for text plus metadata. The difference is within the uncertainty of the test set, so adding metadata does not clearly help.
- At least one of ten data-derived phrases (e.g. "work home", "data entry", "referral bonus") appears in 43.1% of fake postings vs 5.2% of real ones. The phrases were extracted from the full dataset, so the gap is likely overstated; this feature is not used in the models.
- The two most distinctive phrases are company names, which suggests that some fake postings come from recurring sources.
- The missed fake postings (false negatives) do not have more logos or profiles than the typical fake posting; they more often list a salary (39.1% vs 25.8%) and have shorter text. All false positives lack a company profile.
- The class-weighted Logistic Regression over-estimates fraud probabilities over most of the range, so its probabilities are better used for ranking than read as literal rates.

## Limitations

- Only 173 fake postings in the test set: single split, single run, F1 uncertainty of about ±0.04.
- DistilBERT is trained on 68% of the data (validation split), while the baselines use 80%.
- Near-duplicate postings, which the phrase analysis suggests may exist, were not checked for and could make the test results slightly optimistic.
- Business figures rely on illustrative costs.
- The demo serves the Logistic Regression baseline with default values for fields not shown, through a temporary link (one week); it is not a production deployment.

## Tech Stack

- **Language:** Python 3
- **NLP and deep learning:** PyTorch, Hugging Face Transformers (DistilBERT)
- **ML:** scikit-learn, imbalanced-learn (SMOTE), SHAP, LIME
- **Text analysis:** spaCy, TextBlob, textstat
- **Data:** pandas, NumPy, SciPy
- **Visualization:** Matplotlib, Seaborn
- **Demo:** Gradio
- **Environment:** Kaggle notebook (GPU)

Random seeds are fixed (42); small run-to-run differences may remain because of non-deterministic GPU operations.

## Project Structure

```
├── job-posting-fraud-detection.ipynb   # Full pipeline notebook
└── README.md
```

## Notebook

See the notebook for all visualizations, confusion matrices, SHAP plots, ablation results and the business analysis.

[View Notebook on Kaggle](https://www.kaggle.com/code/evdoxialamprinou/job-posting-fraud-detection)

## Author

Evdoxia Lamprinou, MSc AI & Data Science (AUEB)

[LinkedIn](https://www.linkedin.com/in/evdoxia-lamprinou) | [Kaggle Profile](https://www.kaggle.com/evdoxialamprinou)
