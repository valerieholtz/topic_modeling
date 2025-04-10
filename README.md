
# Text Classification for Topic Modelling

## Overview

This project implements a multi-class text classification pipeline using the Reuters-21578 dataset. The original dataset is multi-label, but for the purposes of this study, it wastransformed into a single-label format by selecting the most frequent category for each document.

The goal was to compare the performance of traditional machine learning models with modern transformer-based architectures, focusing particularly on handling high-dimensional, imbalanced, and multi-class data.

---

## Dataset

### Source
- Dataset: [Reuters-21578, Distribution 1.0](http://www.daviddlewis.com/resources/testcollections/reuters21578/)
- Total Documents: 21,578
- Format: SGML
- Labels: Originally multi-label; converted to single-label (most frequent category per document)
- Unique Categories (after filtering): 444

### Preprocessing
- Custom SGML parsing using `lxml` and `ElementTree`
- Lowercasing, tokenization, stopword removal, stemming
- TF-IDF vectorization for classical models
- Label encoding and filtering of rare classes

---

## Models Trained

| Model                 | Accuracy | Macro F1 | Weighted F1 |
|----------------------|----------|----------|-------------|
| Naive Bayes          | 0.42     | 0.02     | 0.34        |
| Random Forest        | 0.43     | 0.06     | 0.37        |
| Tuned Random Forest  | 0.44     | 0.09     | 0.44        |
| XGBoost              | 0.46     | 0.07     | 0.43        |
| BERT (Reuters)       | 0.51     | 0.48     | 0.48        |
| BERT (Fine-tuned on Twitter) | 0.56 | 0.50 | 0.50        |

---

## Label Transformation

- Multi-label → Multi-class using most frequent label
- Tie-breaking handled with random selection
- Invalid, rare, or unstructured labels filtered out
- Labels re-indexed to ensure consistency for BERT

---

## Technical Challenges

- SGML Parsing: Required a custom parser to handle encoding issues and malformed tags.
- Label Range Errors: Mismatched label indices triggered `RuntimeError: CUDA device-side assert`.
- Imbalanced Classes: Most classes had <20 samples; applied class weighting.
- BERT Runtime Issues: CUDA errors traced to Google Colab’s asynchronous GPU handling.
- Fine-tuning: Required early stopping, learning rate warm-up, and output head reshaping for transfer learning.

---

## Domain Generalization

The project further evaluated domain transferability by fine-tuning the Reuters-trained BERT model on a [Twitter vaccine misinformation dataset](https://www.kaggle.com/datasets/prox37/twitter-multilabel-classification-dataset)

### Twitter Dataset Summary
- Samples: 9,921 tweets
- Labels: 12 categories (e.g., side-effect, conspiracy, mandatory)
- Preprocessing: Text cleaning, rare label removal, re-indexing
- Result: F1-score improved to 0.50 post fine-tuning

---

## Future Work

- Transition to multi-label classification using sigmoid outputs
- Try more advanced methods for class imbalance
- Apply active learning to reduce annotation cost
- Experiment with hierarchical classification or label grouping
- Domain-specific pretraining (e.g., news vs. social media)



