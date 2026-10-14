"""Feature engineering for the Titanic dataset."""
import numpy as np
import pandas as pd

TITLE_MAP = {
    "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare", "Capt": "Rare",
    "Sir": "Rare", "Lady": "Rare", "Don": "Rare", "Dona": "Rare",
    "Countess": "Rare", "Jonkheer": "Rare",
}

NUM_COLS = ["Age", "Fare", "FarePerPerson", "FamilySize", "SibSp", "Parch"]
CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Deck", "IsAlone", "HasCabin"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = (
        df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip().map(TITLE_MAP).fillna("Rare")
    )
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("U")  # U = unknown
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    return df


def get_xy(df: pd.DataFrame, with_target: bool = True):
    df = add_features(df)
    X = df[NUM_COLS + CAT_COLS]
    return (X, df["Survived"]) if with_target else X
