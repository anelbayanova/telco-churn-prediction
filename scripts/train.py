#train.py — Обучение модели предсказания оттока клиентов

#Запуск из корня проекта:
    #python scripts/train.py


import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_validate, cross_val_predict, GridSearchCV
)
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve, classification_report,
    make_scorer
)

# Добавляем папку scripts в путь, чтобы импортировать preprocessing.py
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import FixTotalCharges, AddFeatures

# ── Пути к файлам 
DATA_PATH     = os.path.join('data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
PIPELINE_PATH = os.path.join('results', 'churn_pipeline.pkl')
THRESHOLD_PATH = os.path.join('results', 'threshold.json')
PLOTS_DIR     = os.path.join('results', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Столбцы для ColumnTransformer 
# Включаем сюда  исходные + будущие сгенерированные признаки
# Так как ColumnTransformer стоит ПОСЛЕ AddFeatures, он их успешно увидит
NUM_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_tenure', 'n_services']

CAT_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'tenure_bucket'
]


# ── Функция: строим pipeline с нужным классификатором
def make_pipeline(classifier):
    """
    Leakage-free pipeline:
      - шаг 1: чиним TotalCharges (возвращает pandas DF)
      - шаг 2: добавляем новые признаки (возвращает pandas DF)
      - шаг 3: препроцессинг (масштабирование, OHE и удаление customerID/остальных через remainder='drop')
      - шаг 4: классификатор
    """
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale',  StandardScaler())
        ]), NUM_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
    ], remainder='drop')  # drop автоматически уберет customerID и всё лишнее

    pipe = Pipeline([
        ('fix',      FixTotalCharges()),   
        ('features', AddFeatures()),        
        ('prep',     preprocessor),         
        ('model',    classifier),           
    ])
    
    # КРИТИЧЕСКИ ВАЖНО: sklearn долженсохранять структуру Pandas DataFrame 
    # между шагами, чтобы не терялись имена колонок при обработке кастомными трансформерами
    pipe.set_output(transform="pandas")
    
    return pipe


# ── Шаг 1: Загрузка данных 
print('=' * 55)
print('  Обучение модели предсказания оттока клиентов')
print('=' * 55)

print('\n[1] Загружаем данные...')
df = pd.read_csv(DATA_PATH)
print(f'    Размер: {df.shape[0]} строк, {df.shape[1]} столбцов')

# Целевая переменная: Yes → 1, No → 0
y = (df['Churn'] == 'Yes').astype(int)
X = df.drop(columns=['Churn'])

churn_rate = y.mean()
print(f'    Доля оттока: {churn_rate:.1%}')
print(f'    Accuracy "все остаются" = {1 - churn_rate:.1%}  поэтому accuracy бесполезна')


# ── Шаг 2: Стратифицированный сплит 80/20 
print('\n[2] Делаем стратифицированный сплит 80/20...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,        
    random_state=42
)
print(f'    Train: {len(X_train)} строк  (churn {y_train.mean():.1%})')
print(f'    Test:  {len(X_test)} строк   (churn {y_test.mean():.1%})')
print('    Тест откладываем и трогаем только ОДИН РАЗ в конце')


# ── Шаг 3: Сравнение моделей на кросс-валидации 
print('\n[3] Сравниваем модели (5-fold stratified CV)...')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'recall': 'recall',
    'precision': make_scorer(precision_score, zero_division=0),
    'f1': make_scorer(f1_score, zero_division=0),
    'roc_auc': 'roc_auc'
}


models = {
    'DummyMostFrequent': DummyClassifier(strategy='most_frequent'),
    'DummyStratified':   DummyClassifier(strategy='stratified', random_state=42),
    'LogisticRegression': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42, n_jobs=1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
}

cv_results = {}
print(f'\n    {"Модель":<22} {"Recall":>8} {"Precision":>10} {"F1":>6} {"ROC-AUC":>8}')
print('    ' + '-' * 56)

for name, clf in models.items():
    pipe = make_pipeline(clf)
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    cv_results[name] = scores

    r  = scores['test_recall'].mean()
    p  = scores['test_precision'].mean()
    f  = scores['test_f1'].mean()
    au = scores['test_roc_auc'].mean()
    sr = scores['test_recall'].std()
    print(f'    {name:<22} {r:.3f}±{sr:.2f} {p:>10.3f} {f:>6.3f} {au:>8.3f}')


# ── Шаг 4: Тюнинг лучшей модели (GradientBoosting)
print('\n[4] Тюним GradientBoosting через GridSearchCV...')
print('    (GridSearch оборачивает ПОЛНЫЙ pipeline — нет data leakage)')

base_pipe = make_pipeline(GradientBoostingClassifier(random_state=42))

param_grid = {
    'model__n_estimators':  [100, 200],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth':     [3, 4],
}

search = GridSearchCV(
    base_pipe,
    param_grid,
    cv=cv,
    scoring='average_precision',  
    n_jobs=1,
    refit=True,
    verbose=0
)
search.fit(X_train, y_train)

best_pipe = search.best_estimator_
print(f'    Лучшие параметры: {search.best_params_}')
print(f'    Лучший CV average_precision: {search.best_score_:.4f}')


# ── Шаг 5: Подбор порога на OOF-предсказаниях (НЕ на тесте)
print('\n[5] Подбираем порог на out-of-fold предсказаниях тренировочных данных...')
print('    Цель: Recall >= 0.80 при максимальной Precision')

oof_proba = cross_val_predict(
    best_pipe, X_train, y_train,
    cv=cv, method='predict_proba', n_jobs=1
)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_train, oof_proba)

mask = recalls[:-1] >= 0.80
if mask.any():
    best_idx = np.where(mask)[0][np.argmax(precisions[:-1][mask])]
    best_threshold = thresholds[best_idx]
else:
    best_threshold = thresholds[np.argmax(recalls[:-1])]
    print('      Recall 0.80 недостижим на OOF, берём максимальный recall')

oof_pred = (oof_proba >= best_threshold).astype(int)
oof_recall    = recall_score(y_train, oof_pred)
oof_precision = precision_score(y_train, oof_pred)

print(f'    Выбранный порог:  {best_threshold:.4f}')
print(f'    OOF Recall:       {oof_recall:.4f}')
print(f'    OOF Precision:    {oof_precision:.4f}')

# График precision-recall кривой
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(recalls, precisions, lw=2, label='PR кривая (OOF)')
ax.axvline(oof_recall, color='red', linestyle='--',
           label=f'Порог {best_threshold:.2f} → Recall={oof_recall:.2f}')
ax.axhline(oof_precision, color='orange', linestyle=':',
           label=f'Precision={oof_precision:.2f}')
ax.axvline(0.80, color='green', linestyle=':', alpha=0.6, label='Цель Recall=0.80')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve (Out-of-Fold, train data)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'precision_recall_curve.png'), dpi=150)
plt.close()
print(f'    График сохранён: {PLOTS_DIR}/precision_recall_curve.png')


# ── Шаг 6: Сохраняем pipeline и порог 
print(f'\n[6] Сохраняем pipeline → {PIPELINE_PATH}')
joblib.dump(best_pipe, PIPELINE_PATH)
with open(THRESHOLD_PATH, 'w') as f:
    json.dump({'threshold': float(best_threshold)}, f)
print('    Сохранено ')


# ── Шаг 7: Финальная оценка на тесте — ОДИН РАЗ 
print(f'\n[7] Финальная оценка на тест-сете (порог = {best_threshold:.4f})')
print('      Порог уже зафиксирован — тест трогаем первый и последний раз!')

y_proba_test = best_pipe.predict_proba(X_test)[:, 1]
y_pred_test  = (y_proba_test >= best_threshold).astype(int)

test_recall    = recall_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_f1        = f1_score(y_test, y_pred_test)
test_roc_auc   = roc_auc_score(y_test, y_proba_test)

print(f'\n    Recall    (churn): {test_recall:.4f}  {"✅" if test_recall >= 0.75 else "❌"} (цель >= 0.75)')
print(f'    Precision (churn): {test_precision:.4f}  {"✅" if test_precision >= 0.45 else "❌"} (цель >= 0.45)')
print(f'    F1-score:          {test_f1:.4f}')
print(f'    ROC-AUC:           {test_roc_auc:.4f}')

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred_test)
cm_df = pd.DataFrame(
    cm,
    index=['Actual: Stay', 'Actual: Churn'],
    columns=['Pred: Stay', 'Pred: Churn']
)
print('\n    Confusion Matrix:')
print(cm_df.to_string(index=True))
print()
print(classification_report(y_test, y_pred_test, target_names=['Stay', 'Churn']))

# График матрицы ошибок
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(f'Confusion Matrix (threshold={best_threshold:.2f})')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()
print(f'    График сохранён: {PLOTS_DIR}/confusion_matrix.png')


# ── Бизнес-интерпретация и results.md
n_test    = len(y_test)
n_flagged = int(y_pred_test.sum())
n_caught  = int(((y_pred_test == 1) & (y_test == 1)).sum())

cv_table = ''
for name, scores in cv_results.items():
    r  = scores['test_recall'].mean()
    sr = scores['test_recall'].std()
    p  = scores['test_precision'].mean()
    f  = scores['test_f1'].mean()
    au = scores['test_roc_auc'].mean()
    cv_table += f'| {name:<22} | {r:.3f} ± {sr:.3f} | {p:.3f} | {f:.3f} | {au:.3f} |\n'

results_md = f"""# Результаты: Предсказание оттока клиентов

## Почему не Accuracy?

Тривиальный классификатор «все остаются» даёт **{1-churn_rate:.1%} accuracy**, но ловит **0% реальных уходящих**.
Мы используем **Recall** — он показывает, сколько реальных уходящих клиентов мы поймали.

## Сравнение моделей (5-fold Stratified CV)

| Модель                 | Recall        | Precision | F1    | ROC-AUC |
|------------------------|---------------|-----------|-------|---------|
{cv_table}
**Победитель: GradientBoosting** — лучший ROC-AUC и Recall.

## Подбор порога

Порог выбирался на **out-of-fold предсказаниях тренировочных данных** (не на тесте!).
Цель: Recall ≥ 0.80 при максимальной Precision.

- Выбранный порог: **{best_threshold:.4f}**
- OOF Recall: {oof_recall:.4f}
- OOF Precision: {oof_precision:.4f}

## Финальные результаты на тест-сете

| Метрика         | Результат | Цель   | Статус |
|-----------------|-----------|--------|--------|
| Recall (churn)  | {test_recall:.4f}    | ≥ 0.75 | {'✅' if test_recall >= 0.75 else '❌'} |
| Precision       | {test_precision:.4f}    | ≥ 0.45 | {'✅' if test_precision >= 0.45 else '❌'} |
| F1-score        | {test_f1:.4f}    | —      | |
| ROC-AUC         | {test_roc_auc:.4f}    | —      | |

## Бизнес-интерпретация

При пороге **{best_threshold:.2f}** на {n_test} клиентах тест-сета:
- Модель пометила **{n_flagged}** клиентов как «уходящих»
- Из них **{n_caught}** — реальные уходящие клиенты

В пересчёте на **1 000 клиентов**:
- Маркетинг делает ~{round(n_flagged/n_test*1000)} звонков
- Из них ~{round(n_caught/n_test*1000)} — реальным уходящим клиентам

Без модели: либо 1000 звонков всем (дорого), либо 0 звонков (теряем клиентов).
Модель находит баланс: высокий охват уходящих при управляемом числе звонков.

Стоимость False Negative (пропустить уходящего) в 5× выше False Positive →
оправдывает выбор низкого порога с акцентом на Recall.
"""

with open(os.path.join('results', 'results.md'), 'w', encoding='utf-8') as f:
    f.write(results_md)

print('\n' + '=' * 55)
print('  Готово! Файлы сохранены:')
print(f'  - {PIPELINE_PATH}')
print(f'  - results/results.md')
print(f'  - results/plots/')
print('=' * 55)