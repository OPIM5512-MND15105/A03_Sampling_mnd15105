from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def train_evaluate_model(X_train_data, y_train_data, X_test_data, y_test_data, name):
    """Trains a Logistic Regression model and evaluates its performance."""
    print(f"\n--- Training and Evaluating for {name} ---")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_data, y_train_data)
    y_pred = model.predict(X_test_data)

    accuracy = accuracy_score(y_test_data, y_pred)
    report = classification_report(y_test_data, y_pred)

    print(f"Accuracy for {name}: {accuracy:.4f}")
    print(f"Classification Report for {name}:\n{report}")
    return accuracy, y_pred

# Dictionary to store evaluation results and predictions
evaluation_results = {}
y_test_preds = {}

# 1. Evaluate on Original Data
accuracy_original, y_pred_original = train_evaluate_model(X_train, y_train, X_test, y_test, "Original Data")
evaluation_results["Original Data"] = accuracy_original
y_test_preds["Original Data"] = y_pred_original

# 2. Evaluate on Undersampled Data
accuracy_undersampled, y_pred_undersampled = train_evaluate_model(X_resampled_under, y_resampled_under, X_test, y_test, "Undersampled Data")
evaluation_results["Undersampled Data"] = accuracy_undersampled
y_test_preds["Undersampled Data"] = y_pred_undersampled

# 3. Evaluate on Oversampled Data
accuracy_oversampled, y_pred_oversampled = train_evaluate_model(X_resampled_over, y_resampled_over, X_test, y_test, "Oversampled Data")
evaluation_results["Oversampled Data"] = accuracy_oversampled
y_test_preds["Oversampled Data"] = y_pred_oversampled

# 4. Evaluate on SMOTE Data
accuracy_smote, y_pred_smote = train_evaluate_model(X_resampled_smote, y_resampled_smote, X_test, y_test, "SMOTE Data")
evaluation_results["SMOTE Data"] = accuracy_smote
y_test_preds["SMOTE Data"] = y_pred_smote

print("\n--- Summary of Accuracies ---")
for name, acc in evaluation_results.items():
    print(f"{name}: {acc:.4f}")