import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("notebook/Gift_Recommendations_Min5.csv")

X = df[['Gender', 'Age', 'Interest/Hobby']].copy()
y = df['Gift Recommendation']

# Encoders
gender_encoder = LabelEncoder()
hobby_encoder = LabelEncoder()

X['Gender'] = gender_encoder.fit_transform(X['Gender'])
X['Interest/Hobby'] = hobby_encoder.fit_transform(X['Interest/Hobby'])

# Scaler
scaler = StandardScaler()
X['Age'] = scaler.fit_transform(X[['Age']])

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Save everything
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(gender_encoder, open("gender_encoder.pkl", "wb"))
pickle.dump(hobby_encoder, open("hobby_encoder.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model saved successfully!")
