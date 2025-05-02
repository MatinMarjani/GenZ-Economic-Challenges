import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from feature_selection import filter_top_features

data = pd.read_csv('Inflation_Adjusted_Data_Normalized.csv')
X = data.drop(['BIRTHYR', 'AGE', 'YEAR', 'GENERATION'], axis=1)
y = data['GENERATION']

selected_features = filter_top_features(X, data['BIRTHYR'], n_features=20, add_fisher_features=False)
X_selected = X[selected_features]

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Selected features: {selected_features}")
print("Class distribution in the training set before balancing:")
print(y_train.value_counts())

models = {
    'ZeroR': DummyClassifier(strategy="most_frequent"),
    'Random Forest': RandomForestClassifier(n_estimators=1000, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'EBM': ExplainableBoostingClassifier(random_state=42),
    'k-NN': KNeighborsClassifier(n_neighbors=5)
}

results = {}
def plot_confusion_matrix(cm, classes, title):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')

    for i in range(len(cm_normalized)):
        for j in range(len(cm_normalized)):
            color = "white" if cm_normalized[i, j] > 0.5 else "black"
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(j, i, f"{cm_normalized[i, j]:.2f}", ha="center", va="center", color=color, fontsize=10)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=10, ha="center")
    ax.set_yticklabels(classes, fontsize=10, va="center", rotation='vertical')
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title(title, fontsize=16)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

for name, model in models.items():
    print(f"\n{name} Model Evaluation")
    # --- 1. Train-Test Split Evaluation ---
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, test_predictions)
    precision = precision_score(y_test, test_predictions, average='weighted', zero_division=1)
    recall = recall_score(y_test, test_predictions, average='weighted', zero_division=1)
    f1 = f1_score(y_test, test_predictions, average='weighted', zero_division=1)
    cm_test = confusion_matrix(y_test, test_predictions, labels=y.unique())

    results[name] = {
        'Accuracy (80/20 Split)': accuracy,
        'Precision (80/20 Split)': precision,
        'Recall (80/20 Split)': recall,
        'F1-Score (80/20 Split)': f1,
        'Confusion Matrix (Test Set)': cm_test
    }

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")

    plot_confusion_matrix(cm_test, classes=y.unique(), title=f"{name} (80/20 split)")

    # --- 2. Cross-Validation Evaluation (10-fold) ---
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_predictions = cross_val_predict(model, X_selected, y, cv=skf)

    accuracy_cv = accuracy_score(y, cv_predictions)
    precision_cv = precision_score(y, cv_predictions, average='weighted', zero_division=1)
    recall_cv = recall_score(y, cv_predictions, average='weighted', zero_division=1)
    f1_cv = f1_score(y, cv_predictions, average='weighted', zero_division=1)
    cm_cv = confusion_matrix(y, cv_predictions, labels=y.unique())

    results[name].update({
        'Accuracy (CV)': accuracy_cv,
        'Precision (CV)': precision_cv,
        'Recall (CV)': recall_cv,
        'F1-Score (CV)': f1_cv,
        'Confusion Matrix (CV)': cm_cv
    })

    print(f"CV Accuracy: {accuracy_cv:.4f}")
    print(f"CV Precision: {precision_cv:.4f}")
    print(f"CV Recall: {recall_cv:.4f}")
    print(f"CV F1-Score: {f1_cv:.4f}")

    plot_confusion_matrix(cm_cv, classes=y.unique(), title=f"{name} (10-Fold CV)")

print("\nSummary of Results:")
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        if isinstance(value, np.ndarray):
            print(f"{metric}:\n{value}")
        else:
            print(f"{metric}: {value:.4f}")
