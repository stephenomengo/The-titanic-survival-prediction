# Titanic Survival Prediction

Predicting which passengers survived the Titanic using engineered features, a five-model comparison, tuning and stacking. Built on the Kaggle [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) dataset.

**Kaggle score:** 0.77272 (tuned SVM, public leaderboard)

## Approach
1. **EDA**: survival by sex, class, age and missing-value check.
2. **Feature engineering** (`src/features.py`): `Title` from `Name`, `FamilySize`, `IsAlone`, `FarePerPerson`, and `HasCabin` / `Deck` (kept instead of dropping `Cabin`).
3. **Preprocessing**: `ColumnTransformer` with median impute + scale for numeric columns, most-frequent impute + one-hot for categorical columns.
4. **Model comparison** (5-fold stratified CV): Logistic Regression, Random Forest, SVM, HistGradientBoosting, XGBoost.
5. **Tuning**: RandomizedSearchCV on the best model only.
6. **Stacking**: top 3 models with a logistic regression meta-learner.
7. **Evaluation**: hold-out confusion matrix and permutation importance.

## Results
| Model | CV Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| SVM | 0.8294 | 0.7671 | 0.8573 |
| Logistic Regression | 0.8283 | 0.7727 | 0.8719 |
| HistGradientBoosting | 0.8226 | 0.7658 | 0.8811 |
| XGBoost | 0.8181 | 0.7625 | 0.8744 |
| Random Forest | 0.8159 | 0.7571 | 0.8727 |
| **Tuned SVM** | **0.835** | n/a | n/a |
| **Stack (SVM + LogReg + HistGB)** | **0.8474** | 0.7898 | n/a |

## Project structure
```
data/          # put train.csv and test.csv here (not committed)
notebook/      # titanic_survival_prediction.ipynb
src/           # features.py (feature engineering)
submissions/   # generated submission.csv
```

## Run it
```bash
git clone <your-repo-url>
cd titanic-survival-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebook/titanic_survival_prediction.ipynb
```

## Author
Stephen Omengo Owuor
