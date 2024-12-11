import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import f_oneway
import numpy as np


def identify_one_hot_groups(features):
    """
    Identify groups of one-hot-encoded features based on the naming pattern.
    """
    one_hot_groups = {}
    for feature in features:
        if '_' in feature and feature.split('_')[-1].isdigit():  # Identify one-hot encoding pattern
            base_name = '_'.join(feature.split('_')[:-1])
            if base_name not in one_hot_groups:
                one_hot_groups[base_name] = []
            one_hot_groups[base_name].append(feature)
    return one_hot_groups


def fisher_discriminant_score(X, y):
    """
    Compute Fisher Discriminant Score for each feature.
    """
    scores = {}
    for feature in X.columns:
        classes = [X[feature][y == label] for label in np.unique(y)]
        score = f_oneway(*classes).statistic  # Fisher score is based on F-statistic
        scores[feature] = score if not np.isnan(score) else 0
    return scores


def anova_p_values(X, y):
    """
    Compute p-values for each feature using ANOVA F-test.
    """
    p_values = {}
    for feature in X.columns:
        groups = [X[feature][y == label] for label in np.unique(y)]
        _, p_value = f_oneway(*groups)
        p_values[feature] = p_value if not np.isnan(p_value) else 1
    return p_values


def filter_top_features(X, y, n_features, add_fisher_features):
    """
    Filter method to select top features using mutual information, while ensuring
    that one-hot-encoded groups are treated together. Optionally, add 10 new features
    with the highest Fisher scores that are not already selected.
    """
    # mutual information scores
    mi_scores = mutual_info_regression(X, y)
    feature_scores = pd.Series(mi_scores, index=X.columns)
    
    # one-hot-encoded groups
    one_hot_groups = identify_one_hot_groups(X.columns)
    
    # Combine scores for one-hot-encoded groups
    for base_name, group in one_hot_groups.items():
        group_score = feature_scores[group].mean()
        for feature in group:
            feature_scores[feature] = group_score

    # Sort features by MI scores
    sorted_features = feature_scores.sort_values(ascending=False)
    top_features = sorted_features.head(n_features).index.tolist()
    selected_features = set(top_features)
    for base_name, group in one_hot_groups.items():
        if any(feature in top_features for feature in group):
            selected_features.update(group)

    # Fisher scores and ANOVA p-values
    fisher_scores = fisher_discriminant_score(X, y)
    anova_p_vals = anova_p_values(X, y)
    # Sort features by Fisher scores in descending order
    fisher_sorted = sorted(fisher_scores.items(), key=lambda x: x[1], reverse=True)

    # add 10 features with highest Fisher scores not already in selected_features
    additional_features = []
    if add_fisher_features:
        for feature, score in fisher_sorted:
            if feature not in selected_features:
                additional_features.append(feature)
            if len(additional_features) == 10:
                break

    # Combine MI-selected features and optionally Fisher-selected features
    final_features = list(selected_features) + additional_features
    combined_table = []
    for feature in final_features:
        anova_p_value = anova_p_vals.get(feature, "N/A")
        if isinstance(anova_p_value, (float, int)) and anova_p_value < 1e-5:
            anova_p_value = "< 1E-5"
        combined_table.append({
            "Feature": feature,
            "MI Score": feature_scores.get(feature, 0),
            "Fisher Score": fisher_scores.get(feature, 0),
            "ANOVA P-Value": anova_p_value
        })

    # MI-sorted table
    print("\nSelected Features (Ranked by MI Score):")
    print(f"{'Feature':<30} {'MI Score':<10} {'Fisher Score':<15} {'ANOVA P-Value':<15}")
    print("=" * 70)
    for row in sorted(combined_table, key=lambda x: x["MI Score"], reverse=True):
        print(f"{row['Feature']:<30} {row['MI Score']:<10.4f} {row['Fisher Score']:<15.4f} {row['ANOVA P-Value']:<15}")

    # Fisher-sorted table
    print("\nSelected Features (Ranked by Fisher Score):")
    print(f"{'Feature':<30} {'Fisher Score':<15} {'MI Score':<10} {'ANOVA P-Value':<15}")
    print("=" * 70)
    for row in sorted(combined_table, key=lambda x: x["Fisher Score"], reverse=True):
        print(f"{row['Feature']:<30} {row['Fisher Score']:<15.4f} {row['MI Score']:<10.4f} {row['ANOVA P-Value']:<15}")

    return final_features


