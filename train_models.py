import os
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# 1. Generate a compliant dataset (14 features, 1000 instances)
X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, 
                           n_classes=2, random_state=42)

feature_names = [f"Feature_{i+1}" for i in range(14)]
df = pd.DataFrame(X, columns=feature_names)
df['Target'] = y

# Split into Train and Test
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Save test data to CSV as required
test_df.to_csv("test_data.csv", index=False)
print("Saved test_data.csv")

# Separate features and target for training
X_train = train_df.drop(columns=['Target'])
y_train = train_df['Target']

# 2. Define the 6 mandatory models
models = {
    "Logistic_Regression": LogisticRegression(),
    "Decision_Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive_Bayes": GaussianNB(),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# 3. Train and save each model
for name, model in models.items():
    model.fit(X_train, y_train)
    with open(f"model/{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Trained and saved model: model/{name}.pkl")

print("All models trained successfully!")