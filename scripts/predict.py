"""
predict.py
----------
Предсказание оттока для новых клиентов с использованием сохраненного пайплайна.

Запуск из корня репозитория:
    python scripts/predict.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Добавляем scripts/ в PYTHONPATH — нужно для корректной десериализации
# кастомных трансформеров из preprocessing.py
sys.path.insert(0, os.path.dirname(__file__))

# Импортируем модуль явно, чтобы joblib мог найти классы при unpickle
import preprocessing  # noqa: F401

PIPELINE_PATH   = os.path.join("results", "churn_pipeline.pkl")
THRESHOLD_PATH  = os.path.join("results", "threshold.json")
INPUT_PATH      = os.path.join("data", "new_customers.csv")
OUTPUT_PATH     = os.path.join("results", "predictions.csv")


def main():
    print("=" * 50)
    print("  Предсказание оттока для новых клиентов")
    print("=" * 50)

    # --- 1. Загрузка пайплайна ---
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(
            f"Пайплайн не найден: {PIPELINE_PATH}\n"
            "Сначала запустите: python scripts/train.py"
        )
    print(f"\n[1/4] Загрузка пайплайна из {PIPELINE_PATH}")
    pipeline = joblib.load(PIPELINE_PATH)
    print("      Загружен ✅")

    # --- 2. Загрузка порога ---
    if not os.path.exists(THRESHOLD_PATH):
        print(f"      threshold.json не найден → используем порог 0.5 по умолчанию")
        threshold = 0.5
    else:
        with open(THRESHOLD_PATH) as f:
            threshold = json.load(f)["threshold"]
    print(f"[2/4] Используем порог: {threshold:.4f}")

    # --- 3. Загрузка новых клиентов ---
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Файл новых клиентов не найден: {INPUT_PATH}")
    print(f"[3/4] Загрузка новых клиентов из {INPUT_PATH}")
    df_new = pd.read_csv(INPUT_PATH)
    print(f"      Загружено строк: {len(df_new)}")

    # Откладываем customerID в сторону
    if "customerID" in df_new.columns:
        customer_ids = df_new["customerID"].copy()
    else:
        customer_ids = pd.Series(range(len(df_new)), name="customerID")

    # Формируем матрицу признаков X_new. 
    # Убираем Churn и customerID, чтобы состав колонок на входе 
    # строго соответствовал тому, что ожидает пайплайн (опция remainder='drop' их тоже съест,
    # но  лучше убрать их превентивно )
    X_new = df_new.drop(columns=["Churn", "customerID"], errors="ignore")

    # --- 4. Предсказание ---
    print("[4/4] Предсказание...")
    
    churn_proba = pipeline.predict_proba(X_new)[:, 1]
    churn_pred  = (churn_proba >= threshold).astype(int)

    # Сборка результата
    predictions = pd.DataFrame({
        "customerID":  customer_ids.values,
        "churn_pred":  churn_pred,
        "churn_proba": np.round(churn_proba, 4),
    })

    # Сохранение
    os.makedirs("results", exist_ok=True)
    predictions.to_csv(OUTPUT_PATH, index=False)

    print(f"\n      Результаты сохранены: {OUTPUT_PATH}")
    print(f"\n      Из {len(predictions)} клиентов:")
    print(f"      - Помечены как уходящие (churn_pred=1): {churn_pred.sum()}")
    print(f"      - Остаются (churn_pred=0):               {(churn_pred==0).sum()}")

    print("\n--- Детали предсказаний (первые 20 строк) ---")
    print(predictions.head(20).to_string(index=False))

    print("\n" + "=" * 50)
    print("  Готово!")
    print("=" * 50)


if __name__ == "__main__":
    main()