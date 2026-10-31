from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset" / "SMSSpamCollection"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "sms_model.pkl"

def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Download the UCI SMS Spam Collection dataset, extract it, "
            "and place the file named 'SMSSpamCollection' in the dataset folder."
        )

    data = pd.read_csv(
        DATASET_PATH,
        sep="\\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
        on_bad_lines="skip"
    )

    data = data.dropna(subset=["label", "message"])
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data = data[data["label"].isin(["ham", "spam"])]
    data["target"] = data["label"].map({"ham": 0, "spam": 1})

    if data.empty or data["target"].nunique() < 2:
        raise ValueError("Dataset must contain both 'ham' and 'spam' messages.")

    X = data["message"].astype(str)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )),
        ("classifier", MultinomialNB(alpha=0.5))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Dataset rows: {len(data)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, target_names=["Normal (ham)", "Spam"]
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nTrained model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
