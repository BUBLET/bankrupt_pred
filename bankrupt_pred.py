import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import klib
import wget
import tensorflow as tf
from tensorflow.keras import layers


helper_functions_file = "helper_functions.py"

# Шаг 1: Проверяем, существует ли файл helper_functions.py
if not os.path.isfile(helper_functions_file):
    # Скачиваем файл helper_functions.py, если его нет
    wget.download("https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py")

# Импортируем функции из файла helper_functions.py
from helper_functions import calculate_results

# Шаг 2: Импорт данных
df = pd.read_csv("bankrupt_data.csv")

## Шаг 3: Очистка данных 
df_clean = klib.data_cleaning(df)

df_clean.to_csv("clean_data.csv", index=False)
# Шаг 4: Проверка на несбалансированные данные

# print(df_clean['bankrupt'].value_counts())

# Шаг 5: Преобразование несбалансированных данных
X = df_clean.drop('bankrupt', axis='columns')
y = df_clean['bankrupt']
smote = SMOTE(sampling_strategy ='minority')
X, y = smote.fit_resample(X,y)


# Шаг 6: Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 7: Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Шаг 8: Обучение моделей

# Модель 1: Логическая регрессия
logreg_model = LogisticRegression(max_iter = 10000)
logreg_model.fit(X_train_scaled, y_train)
y_probs = logreg_model.predict(X_test_scaled)
y_test_rounded = np.round(y_test)

# Преобразуем предсказанные вероятности в бинарные классификации
y_pred_bin = np.where(y_probs > 0.5, 1, 0)

# Оценка результатов и вычисление метрик
logistic = calculate_results(y_test_rounded, y_pred_bin)

# Модель 2: Глубокая нейронная сеть
tf_model = tf.keras.Sequential([
    layers.Dense(64, activation = 'relu'),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(512, activation = 'relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation = 'sigmoid')
])
tf_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = tf_model.fit(X_train, y_train, validation_split = 0.2, epochs = 20, verbose = 0)
#plot_loss_curves(history)

y_probs = tf_model.predict(X_test_scaled)
y_test_rounded = np.round(y_test)

# Преобразуем предсказанные вероятности в бинарные классификации
y_pred_bin = np.where(y_probs > 0.5, 1, 0)

# Оценка результатов и вычисление метрик
dnn = calculate_results(y_test_rounded, y_pred_bin)

# Сохранение модели
tf_model.save("bankrupt_pred.keras")

#Вывод итогов
model_perf = pd.DataFrame({"Logistic Regression":logistic,"Deep Neural Network":dnn})
print(model_perf)