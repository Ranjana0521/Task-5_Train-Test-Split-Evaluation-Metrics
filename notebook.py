import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

df = pd.read_csv("Heart_disease_dataset.csv")
df.head()
print(df)

#1.Split dataset into train and test sets.
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#3.Train a simple model (Logistic Regression).
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#4.Predict on test data.
y_pred = model.predict(X_test)

print("Predicted:", y_pred)
print("Actual:", y_test.values)

#5.Calculate accuracy, precision, recall.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

#6.Confusion matrix.
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Deliverables trained model(Prediction)
output = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
output.to_csv("predictions.csv", index=False)

#%%
#Deliverables evaluation_report
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, zero_division=0)
file = open("evaluation_report.txt", "w")
file.write("Accuracy: " + str(accuracy) + "\n")
file.write("Precision: " + str(precision) + "\n")
file.write("Recall: " + str(recall) + "\n")
file.write("Confusion Matrix:\n" + str(cm))
file.close()

print("Evaluation report saved!")
