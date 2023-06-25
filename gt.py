from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Загрузка набора данных Iris
iris_data = load_iris()
X = iris_data.data
y = iris_data.target

# Разделение набора данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X_train, y_train)

# Прогнозирование классов на тестовой выборке
y_pred = model.predict(X_test)

# Вычисление точности модели
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
