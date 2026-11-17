import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#Load Training dataset
data = pd.read_csv("titanic/train.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

#Select useful features
features = [
"Pclass",
"Sex",
"Age",
"SibSp",
"Parch",
"Fare",
"Embarked"
]

X = data[features]
y = data["Survived"]

#Identifyiing numerical and categorical columns
numerical_features = [
"Pclass", "Age", "SibSp", "Parch", "Fare"
]

categorical_features = ["Sex", "Embarked"]

#Preprocessing numerical data
numerical_pipeline = Pipeline([
("imputer", SimpleImputer(strategy="median")),
("scaler", StandardScaler())
])

#Preprocessing categorical data
categorical_pipeline = Pipeline([
("imputer", SimpleImputer(strategy="most_frequent")),
("encoder", OneHotEncoder(handle_unknown="ignore"))
])

#Final processing
preprocessor = ColumnTransformer([
("num", numerical_pipeline, numerical_features),
("cat", categorical_pipeline, categorical_features)
])

#ML Pipeline
model = Pipeline([
("preprocessor", preprocessor),
("classifier", RandomForestClassifier(
n_estimators=100,
random_state=42
))
])

#Split dataset into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

#Train Model
model.fit(X_train, y_train)

print("\nModel training completed!")

#Evaluate Model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, predictions))

#Save Trained Model
joblib.dump(model, "model.pkl")

print("\nTrained model saved as model.pkl")
