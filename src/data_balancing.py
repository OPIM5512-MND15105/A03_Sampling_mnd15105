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

# Initialize dictionaries to store accuracies for each run
all_accuracies = {
    "Original Data": [],
    "Undersampled Data": [],
    "Oversampled Data": [],
    "SMOTE Data": []
}

num_runs = 30
print(f"\n--- Running {num_runs} iterations for reproducibility ---")

for i in range(num_runs):
    print(f"\nIteration {i+1}/{num_runs}")
    try:
        # 1. Evaluate on Original Data
        accuracy_original, _ = train_evaluate_model(X_train, y_train, X_test, y_test, "Original Data")
        all_accuracies["Original Data"].append(accuracy_original)

        # 2. Evaluate on Undersampled Data
        accuracy_undersampled, _ = train_evaluate_model(X_resampled_under, y_resampled_under, X_test, y_test, "Undersampled Data")
        all_accuracies["Undersampled Data"].append(accuracy_undersampled)

        # 3. Evaluate on Oversampled Data
        accuracy_oversampled, _ = train_evaluate_model(X_resampled_over, y_resampled_over, X_test, y_test, "Oversampled Data")
        all_accuracies["Oversampled Data"].append(accuracy_oversampled)

        # 4. Evaluate on SMOTE Data
        accuracy_smote, _ = train_evaluate_model(X_resampled_smote, y_resampled_smote, X_test, y_test, "SMOTE Data")
        all_accuracies["SMOTE Data"].append(accuracy_smote)
    except NameError as e:
        print(f"Error: {e}. Please ensure all preceding cells have been executed.")
        break # Exit loop if essential variables are not defined

import numpy as np

print(f"\n--- Summary of Accuracies over {num_runs} Runs ---")
for name, accuracies in all_accuracies.items():
    if accuracies:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"{name}: Mean Accuracy = {mean_acc:.4f}, Std Dev = {std_acc:.4f}")
    else:
        print(f"{name}: No accuracy data available.")
print("\n--- Summary of Accuracies ---")
for name, acc in evaluation_results.items():
    print(f"{name}: {acc:.4f}")