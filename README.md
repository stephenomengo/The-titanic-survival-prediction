# Titanic Survival Prediction

Predicting which passengers survived the Titanic using engineered features, a five-model comparison, tuning and stacking. Built on the Kaggle [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) dataset.

**Kaggle score:** 

## Approach
1. **EDA**: survival by sex, class, age and missing-value check.
2. **Feature engineering** (`src/features.py`): `Title` from `Name`, `FamilySize`, `IsAlone`, `FarePerPerson`, and `HasCabin` / `Deck` (kept instead of dropping `Cabin`).
3. **Preprocessing**: `ColumnTransformer` with median impute + scale for numeric columns, most-frequent impute + one-hot for categorical columns.
4. **Model comparison** (5-fold stratified CV): Logistic Regression, Random Forest, SVM, HistGradientBoosting, XGBoost.
5. **Tuning**: RandomizedSearchCV on the best model only.
6. **Stacking**: top 3 models with a logistic regression meta-learner.
7. **Evaluation**: hold-out confusion matrix and permutation importance.

## Results
| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| _fill from notebook section 4_ | | | |

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
