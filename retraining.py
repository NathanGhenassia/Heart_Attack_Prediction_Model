import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("data/heart.csv")

y = df["output"]
X = df.drop(["output"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

param_grid_logreg = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2']
}

grid_search_log_reg = GridSearchCV(pipeline, param_grid_logreg, cv=5, n_jobs=-1)
grid_search_log_reg.fit(X_train, y_train)

best_params = grid_search_log_reg.best_params_
best_model = grid_search_log_reg.best_estimator_
coefficients = best_model.named_steps['classifier'].coef_[0]
feature_names = X_train.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefficients, color='skyblue')
plt.xlabel('Coeficiente')
plt.title('Importancia de las características en el modelo de regresión logística')
plt.savefig("feature_importance.png", dpi=120)
plt.close()

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

with open("metrics.txt", 'w') as outfile:
    outfile.write("Training accuracy: %2.1f%%\n" % accuracy)

model_filename = 'model/logistic_regression_model_heart_attack_v01.pkl'
joblib.dump(best_model, model_filename)
