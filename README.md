# Job Posting Fraud Detection

End-to-end NLP fraud detection pipeline built on 17,880 real job postings, combining TF-IDF baselines, DistilBERT fine-tuning, SHAP explainability, and a live Gradio demo.

## Business Problem

Online job platforms receive thousands of new postings daily. Fraudulent listings — designed to collect personal information, extract money, or deceive job seekers — are difficult to detect manually at scale. This project builds an automated NLP pipeline that classifies job postings as real or fraudulent, enabling platforms to flag suspicious content in real time, protect users from scams, and reduce moderation costs.

A cost-benefit analysis shows that the model reduces fraud-related cost by **84.2%** compared to a no-model baseline — and when scaled to a platform processing 1 million postings, it would intercept over 41,000 fraudulent listings that would otherwise go undetected.

## Dataset

- **Source:** [Real / Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) (Kaggle)
- **Size:** 17,880 job postings
- **Target:** Binary — Real (95.2%) vs Fake (4.8%)
- **Challenge:** Severe class imbalance and high structural similarity between real and fake postings — fake listings deliberately mimic legitimate ones to avoid detection

## Approach

| Step | Description |
|------|-------------|
| EDA | Class imbalance analysis, text length distributions, word clouds, categorical patterns across employment type, experience level, and company branding |
| Preprocessing | HTML cleaning, lowercasing, stopword removal, unified text field combining title, description, requirements, and company profile |
| Feature Engineering | 13 structural and text-derived features including has_company_logo, has_company_profile, word_count, and data-driven urgency phrase detection |
| Baseline Models | TF-IDF (ngram 1-3) combined with Logistic Regression, Random Forest, and LinearSVC |
| Imbalanced Data Handling | Three techniques compared — class weighting, SMOTE, and threshold tuning — to maximize fraud detection Recall |
| Advanced Model | DistilBERT fine-tuning with early stopping, class-weighted CrossEntropyLoss, and HuggingFace Transformers |
| Explainability | SHAP global feature importance and LIME local explanations per prediction |
| Advanced NLP | Data-driven urgency phrase detection via frequency ratio analysis, Named Entity Recognition (NER), sentiment polarity, and readability scoring |
| Ablation Study | Systematic comparison of text only vs metadata only vs title only vs description only vs all combined |
| Cross-validation | Stratified K-Fold (5 folds), calibration curve analysis, and bootstrap 95% confidence intervals |
| Error Analysis | False negative and false positive profiling to identify systematic model failures |
| Business Framing | Cost-benefit analysis with asymmetric error costs, scaled impact simulation to 1M postings |
| Deployment | Live Gradio interactive demo — input any job posting and receive a fraud probability score |

## Key Findings

- `has_company_logo` and `has_company_profile` are the strongest structural fraud signals — fake postings lack both at rates far above real ones (32.7% vs 81.9% for logo; 32.2% vs 84.0% for profile)
- Data-driven urgency phrases such as "work home", "data entry", and "referral bonus" appear in **27.4%** of fake postings vs only **2.8%** of real ones — a near 10x ratio
- Text alone drives the majority of predictive power (F1: 0.833) — metadata adds meaningful Recall but text is the dominant signal
- t-SNE visualization confirms that fake postings do not form a distinct cluster in TF-IDF embedding space, explaining why contextual models like DistilBERT outperform bag-of-words approaches
- False negatives are systematically harder to catch — they possess company logos and profiles at rates far above the typical fake posting, effectively mimicking legitimate listings

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| Logistic Regression | 0.77 | 0.87 | 0.81 | 0.991 |
| Random Forest | 1.00 | 0.59 | 0.74 | 0.990 |
| LinearSVC | 0.96 | 0.76 | 0.85 | 0.991 |
| **DistilBERT** | **0.90** | **0.86** | **0.88** | **0.989** |

DistilBERT is selected as the final model, delivering the best F1 (0.88) and the most balanced Precision/Recall trade-off. In fraud detection, Recall is the primary metric — a fraudulent posting that escapes detection is far more costly than a real posting incorrectly flagged.

## Tech Stack

- **Language:** Python 3
- **NLP & Deep Learning:** HuggingFace Transformers, DistilBERT, PyTorch
- **ML:** Scikit-learn, SHAP, LIME
- **Data:** Pandas, NumPy, SciPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Gradio
- **Environment:** Kaggle (GPU T4 x2)

## Project Structure
├── job-posting-fraud-detection.ipynb   # Full pipeline notebook
└── README.md


## Results & Notebook

See the full notebook for all visualizations, confusion matrices, SHAP plots, ablation results, and business recommendations.

[View Notebook on Kaggle](https://www.kaggle.com/code/evdoxialamprinou/job-posting-fraud-detection)

## Author

Evdoxia Lamprinou — MSc AI & Data Science

[LinkedIn](https://www.linkedin.com/in/evdoxia-lamprinou) | [Kaggle Profile](https://www.kaggle.com/evdoxialamprinou)
