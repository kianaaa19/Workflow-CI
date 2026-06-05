import os
import pandas as pd
from xgboost import XGBClassifier
import mlflow

# Izinkan file store jika terpaksa lari ke lokal
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

def main():
    # SOLUSI MUTAKHIR: Ambil tracking URI dari Environment Variable (DagsHub).
    # Jika ENV kosong (pas lu run manual di terminal), baru dia otomatis pakai lokal "file:./mlruns"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment("Bank_Churn_Optimization_Santanam")

    # Load data hasil preprocessing (Gunakan path relatif yang aman)
    train = pd.read_csv("churn_bank_ABC_preprocessing/train_preprocessed.csv")
    test = pd.read_csv("churn_bank_ABC_preprocessing/test_preprocessed.csv")

    X_train, y_train = train.drop(columns=['churn']), train['churn']
    X_test, y_test = test.drop(columns=['churn']), test['churn']

    mlflow.autolog()
    print(f"[+] MLflow Tracking URI aktif ke: {tracking_uri}")

    with mlflow.start_run(run_name="XGBoost_Baseline_santanam"):
        model = XGBClassifier(random_state=42, eval_metric='logloss')
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        print(f"[+] Eksekusi Baseline Sukses. Akurasi: {acc}")

if __name__ == "__main__":
    main()
