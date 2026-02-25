from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


def train_evaluate_model(X_train, y_train, X_test, y_test, label):
    """Trains a Logistic Regression model and evaluates its performance."""

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n--- {label} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")

    return accuracy, y_pred


def main():
    """
    Main execution block.
    NOTE:
    This assumes X_train, y_train, X_test, y_test,
    X_resampled_under, y_resampled_under,
    X_resampled_over, y_resampled_over,
    X_resampled_smote, y_resampled_smote
    are already defined above this function
    (or imported from another module).
    """

    # Dictionary to store evaluation results and predictions
    evaluation_results = {}
    y_test_preds = {}

    # 1. Original Data
    accuracy_original, y_pred_original = train_evaluate_model(
        X_train, y_train, X_test, y_test, "Original Data"
    )
    evaluation_results["Original Data"] = accuracy_original
    y_test_preds["Original Data"] = y_pred_original

    # 2. Undersampled Data
    accuracy_under, y_pred_under = train_evaluate_model(
        X_resampled_under, y_resampled_under, X_test, y_test, "Undersampled Data"
    )
    evaluation_results["Undersampled Data"] = accuracy_under
    y_test_preds["Undersampled Data"] = y_pred_under

    # 3. Oversampled Data
    accuracy_over, y_pred_over = train_evaluate_model(
        X_resampled_over, y_resampled_over, X_test, y_test, "Oversampled Data"
    )
    evaluation_results["Oversampled Data"] = accuracy_over
    y_test_preds["Oversampled Data"] = y_pred_over

    # 4. SMOTE Data
    accuracy_smote, y_pred_smote = train_evaluate_model(
        X_resampled_smote, y_resampled_smote, X_test, y_test, "SMOTE Data"
    )
    evaluation_results["SMOTE Data"] = accuracy_smote
    y_test_preds["SMOTE Data"] = y_pred_smote

    # Reproducibility runs
    all_accuracies = {
        "Original Data": [],
        "Undersampled Data": [],
        "Oversampled Data": [],
        "SMOTE Data": [],
    }

    num_runs = 30
    print(f"\n--- Running {num_runs} iterations ---")

    for i in range(num_runs):
        print(f"\nIteration {i + 1}/{num_runs}")

        acc, _ = train_evaluate_model(X_train, y_train, X_test, y_test, "Original Data")
        all_accuracies["Original Data"].append(acc)

        acc, _ = train_evaluate_model(X_resampled_under, y_resampled_under, X_test, y_test, "Undersampled Data")
        all_accuracies["Undersampled Data"].append(acc)

        acc, _ = train_evaluate_model(X_resampled_over, y_resampled_over, X_test, y_test, "Oversampled Data")
        all_accuracies["Oversampled Data"].append(acc)

        acc, _ = train_evaluate_model(X_resampled_smote, y_resampled_smote, X_test, y_test, "SMOTE Data")
        all_accuracies["SMOTE Data"].append(acc)

    print(f"\n--- Summary of Accuracies over {num_runs} Runs ---")
    for name, accuracies in all_accuracies.items():
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"{name}: Mean Accuracy = {mean_acc:.4f}, Std Dev = {std_acc:.4f}")

    print("\n--- Single Run Summary ---")
    for name, acc in evaluation_results.items():
        print(f"{name}: {acc:.4f}")


if __name__ == "__main__":
    main()
