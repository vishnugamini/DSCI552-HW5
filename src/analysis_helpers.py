"""Shared preprocessing, rule formatting, and figures for Homework 5."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from IPython.display import Markdown, display
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree


def decision_rules(model, feature_names, class_names):
    tree, output = model.tree_, []

    def visit(node, conditions):
        left, right = tree.children_left[node], tree.children_right[node]
        if left == right:
            predicted = int(model.classes_[np.argmax(tree.value[node][0])])
            output.append({"leaf": node, "IF": " AND ".join(conditions) or "TRUE",
                           "THEN": class_names[predicted], "joint_class": predicted,
                           "records": int(tree.n_node_samples[node])})
            return
        name = feature_names[tree.feature[node]]
        threshold = float(tree.threshold[node])
        if name != "temperature" and np.isclose(threshold, .5):
            low, high = f"{name} = no", f"{name} = yes"
        else:
            low, high = f"{name} <= {threshold:.2f}", f"{name} > {threshold:.2f}"
        visit(left, conditions + [low])
        visit(right, conditions + [high])

    visit(0, [])
    return pd.DataFrame(output)


def show_rules(rules):
    lines = [f"{i+1}. IF {row['IF']}, THEN **{row['THEN']}**."
             for i, (_, row) in enumerate(rules.iterrows())]
    display(Markdown("\n".join(lines)))


def preprocessing(standardize=False):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler() if standardize else "passthrough")])


def regression_pipeline(estimator, standardize=False):
    return Pipeline([("preprocess", preprocessing(standardize)), ("model", estimator)])


def make_fold_cache(X, y, folds):
    cache = []
    for train, valid in folds:
        raw = preprocessing(False).fit(X.iloc[train])
        scaled = preprocessing(True).fit(X.iloc[train])
        cache.append({"raw_train": raw.transform(X.iloc[train]),
                      "raw_valid": raw.transform(X.iloc[valid]),
                      "scaled_train": scaled.transform(X.iloc[train]),
                      "scaled_valid": scaled.transform(X.iloc[valid]),
                      "y_train": y.iloc[train].to_numpy(),
                      "y_valid": y.iloc[valid].to_numpy()})
    return cache


def cv_results_frame(search, parameter, label):
    scores = search.cv_results_
    return pd.DataFrame({label: scores['param_' + parameter].astype(float),
                         'cv_mse': -scores['mean_test_score'],
                         **{f'fold_{i+1}': -scores[f'split{i}_test_score'] for i in range(5)}})


def save_details(models, columns, directory, tree_cv, full_tree, pruned_tree,
                 top_features, convergence_messages):
    coefficients = pd.DataFrame(index=columns)
    for name in ['OLS', 'Ridge standardized', 'LASSO original', 'LASSO standardized']:
        coefficients[name] = models[name].named_steps['model'].coef_
    coefficients.to_csv(directory / 'regression_coefficients.csv', index_label='feature')
    best = tree_cv.loc[tree_cv.cv_accuracy.idxmax()]
    one_se = tree_cv.loc[tree_cv.cv_accuracy >= best.cv_accuracy-best.cv_se-1e-12]
    one_se = one_se.sort_values('alpha').iloc[-1]
    summary = {
        'tree_one_se_alpha': float(one_se.alpha),
        'tree_compact_alpha': float(pruned_tree.ccp_alpha),
        'tree_full_leaves': int(full_tree.get_n_leaves()),
        'tree_compact_leaves': int(pruned_tree.get_n_leaves()),
        'ridge_lambda': float(models['Ridge standardized'].named_steps['model'].alpha),
        'lasso_original_lambda': float(models['LASSO original'].named_steps['model'].alpha),
        'lasso_standardized_lambda': float(models['LASSO standardized'].named_steps['model'].alpha),
        'lasso_original_selected': list(columns[np.abs(coefficients['LASSO original']) > 1e-8]),
        'lasso_standardized_selected': list(columns[np.abs(coefficients['LASSO standardized']) > 1e-8]),
        'pcr_M': int(models['PCR standardized'].named_steps['pca'].n_components),
        'xgboost_alpha': float(models['XGBoost L1 trees'].named_steps['model'].reg_alpha),
        'top_cv_features': top_features,
        'convergence_warnings': convergence_messages,
    }
    (directory / 'selected_models.json').write_text(json.dumps(summary, indent=2)+'\n')


def show_tree(model, feature_names, class_names, title, size):
    fig, ax = plt.subplots(figsize=size)
    short_names = [f"B{k//2}/N{k%2}" for k in range(len(class_names))]
    plot_tree(model, feature_names=list(feature_names), class_names=short_names,
              filled=True, rounded=True, impurity=False, fontsize=12, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def show_pruning(table, alpha):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].step(table.alpha, table.leaves, where="post")
    axes[0].set(xlabel="Cost-complexity alpha", ylabel="Leaves", title="Subtree size")
    axes[1].errorbar(table.alpha, table.cv_accuracy, yerr=table.cv_se, marker=".", capsize=2)
    axes[1].axhline(.80, color="#aa4433", linestyle="--", label="80% floor")
    axes[1].axvline(alpha, color="#228833", linestyle=":", label="Compact choice")
    axes[1].set(xlabel="Cost-complexity alpha", ylabel="Mean validation accuracy")
    axes[1].legend()
    fig.suptitle("Figure 2. Cost-complexity pruning")
    fig.tight_layout()
    plt.show()


def show_correlations(correlations):
    fig, ax = plt.subplots(figsize=(20, 18))
    image = ax.imshow(correlations, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    ax.set_xticks(np.arange(len(correlations)), correlations.columns, rotation=90, fontsize=5)
    ax.set_yticks(np.arange(len(correlations)), correlations.index, fontsize=5)
    fig.colorbar(image, ax=ax, fraction=.035, pad=.015, label="Pearson correlation")
    ax.set_title("Figure 4. Correlations among the 122 training predictors")
    fig.tight_layout()
    plt.show()


def show_exploration(X, y, columns):
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for column, ax in zip(columns, axes.flat):
        ax.scatter(X[column], y, s=8, alpha=.3, color="#286a9e")
        ax.set(xlabel=column, ylabel=y.name)
    axes.flat[-1].set_visible(False)
    fig.suptitle("Figure 5. Highest-CV predictors versus the response, training data")
    fig.tight_layout()
    plt.show()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.boxplot([X[c] for c in columns], labels=columns,
               flierprops={"markersize": 2, "alpha": .3})
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Normalized predictor value")
    ax.set_title("Figure 6. Distributions of the 11 highest-CV training predictors")
    fig.tight_layout()
    plt.show()


def show_cv_curves(ridge, ridge_lambda, lassos, models, pcr, M, xgb, alpha, comparison):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].semilogx(ridge["lambda"], ridge.cv_mse)
    axes[0, 0].axvline(ridge_lambda, linestyle="--", color="#aa4433")
    axes[0, 0].set(title="Ridge", xlabel="Lambda", ylabel="Validation MSE")
    for ax, label in zip(axes[0, 1:], ["LASSO original", "LASSO standardized"]):
        ax.semilogx(lassos[label]["lambda"], lassos[label].cv_mse)
        ax.axvline(models[label].named_steps["model"].alpha, linestyle="--", color="#aa4433")
        ax.set(title=label, xlabel="Lambda", ylabel="Validation MSE")
    axes[1, 0].plot(pcr.M, pcr.cv_mse)
    axes[1, 0].axvline(M, linestyle="--", color="#aa4433")
    axes[1, 0].set(title="PCR", xlabel="Principal components", ylabel="Validation MSE")
    axes[1, 1].plot(xgb.alpha, xgb.cv_mse, marker=".")
    axes[1, 1].set_xscale("symlog", linthresh=.001)
    axes[1, 1].axvline(alpha, linestyle="--", color="#aa4433")
    axes[1, 1].set(title="XGBoost", xlabel="L1 alpha", ylabel="Validation MSE")
    axes[1, 2].barh(comparison.index, comparison.test_mse, color="#3974a6")
    axes[1, 2].set(xlabel="Test MSE", title="Held-out errors")
    fig.suptitle("Figure 7. Cross-validation curves and held-out errors")
    fig.tight_layout()
    plt.show()
