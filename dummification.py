import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------------
# CREATE SAMPLE DATA
# --------------------------
data = {
    'Color': ['Green', 'Red', 'Blue', 'Green', 'Red', 'Blue'],
    'Size': ['Small', 'Large', 'Medium', 'Medium', 'Small', 'Large'],
    'Label': [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# --------------------------
# FEATURE DUMMIFICATION (ONE-HOT ENCODING)
# --------------------------
df_encoded = pd.get_dummies(df, columns=['Color', 'Size'], drop_first=True)
print("\nDataFrame after Feature Dummification:")
print(df_encoded)

# --------------------------
# SPLIT DATA INTO TRAIN AND TEST
# --------------------------
X = df_encoded.drop('Label', axis=1)
y = df_encoded['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# TRAIN RANDOM FOREST CLASSIFIER
# --------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --------------------------
# PREDICTION AND EVALUATION
# --------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
