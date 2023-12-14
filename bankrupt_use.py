import tensorflow as tf
import pandas as pd

# Загрузка сохраненной модели
trained_model = tf.keras.models.load_model("bankrupt_pred.keras")

# Предсказание на новых данных
df = pd.read_csv("clean_data.csv")
X = df.drop('bankrupt', axis='columns')

predictions = trained_model.predict(X)

# Вывод результатов предсказания
print(predictions)