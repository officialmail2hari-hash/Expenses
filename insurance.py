import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# Medical Cost Personal Insurance dataset (Kaggle)
df = pd.read_csv('D:\Harishma M-2408229\Dataset\insurance.csv', encoding="latin")
print("Dataset sample:\n", df.head())
print("\nShape:", df.shape)
# Encode categorical columns: sex, smoker, region
df_encoded = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)
print("\nEncoded columns:\n", df_encoded.columns.tolist())

X = df_encoded.drop("expenses", axis=1)
y = df_encoded["expenses"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
import matplotlib.pyplot as plt
# Actual vs Predicted Scatter Plot
plt.figure(figsize=(7,6))
# Blue dots
plt.scatter(y_test, y_pred, color="blue", label="Data Points")
# Straight Regression / Reference Line
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color="red",
    linewidth=2,
    label="Regression Line"
)
plt.xlabel("Actual Expenses")
plt.ylabel("Predicted Expenses")
plt.title("Actual vs Predicted Insurance Charges")
plt.legend()
plt.grid(True)
plt.show()
print("\nCoefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.2f}")
print("Intercept:", model.intercept_)
print("\nR2 Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
# Predict for a new person: age=35, bmi=28, children=2, sex_male=1, smoker_yes=0, region_southeast=1
new_person = pd.DataFrame([{ "age": 35, "bmi": 28, "children": 2,  "sex_male": 1, "smoker_yes": 0,    "region_northwest": 0, "region_southeast": 1, "region_southwest": 0}])
new_person = new_person[X.columns]  # match column order
print("\nPredicted charges:", model.predict(new_person)[0])
