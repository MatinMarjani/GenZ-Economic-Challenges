import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

selected_features = filter_top_features(X, data['BIRTHYR'], n_features=20, add_fisher_features= False)  # Adjust n_features as needed
X_selected = X[selected_features]

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

print(f"Selected features: {selected_features}")
print("Class distribution in the training set before balancing:")
print(y_train.value_counts())

models = {
    'ZeroR (Baseline)': DummyClassifier(strategy="most_frequent"),  
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
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=1)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=1)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=1)
    cm = confusion_matrix(y_test, predictions, labels=y.unique())

    results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'Confusion Matrix': cm}

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

    for i in range(len(cm)):
        for j in range(len(cm)):
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black", fontsize=10)

    class_names = y.unique()
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=10, ha="center")
    ax.set_yticklabels(class_names, fontsize=10, va="center", rotation='vertical')
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title(f"Confusion Matrix for {name}", fontsize=16)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

for name, metrics in results.items():
    print(f"\n{name} Results:")
    for metric, value in metrics.items():
        if metric == 'Confusion Matrix':
            print(f"{metric}:\n{value}")
        else:
            print(f"{metric}: {value:.4f}")
