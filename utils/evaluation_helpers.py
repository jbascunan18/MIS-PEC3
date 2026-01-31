import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import (
    ShuffleSplit,
    GridSearchCV,
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

import itertools


def plot_confusion_matrix(cm, classes, title="Confusion matrix", cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.show()


def train_and_evaluate_classifier(X, yt, estimator, grid):
    # Cross-validation over selected estimator
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    classifier = GridSearchCV(
        estimator=estimator,
        cv=cv,
        param_grid=grid,
        error_score=0.0,
        n_jobs=-1,
        verbose=5,
    )

    # Training
    print("Training model")
    classifier.fit(X, yt)

    # Cross-validation results
    print("CV-scores for each grid configuration")
    means = classifier.cv_results_["mean_test_score"]
    stds = classifier.cv_results_["std_test_score"]
    for mean, std, params in sorted(
        zip(means, stds, classifier.cv_results_["params"]), key=lambda x: -x[0]
    ):
        print("Accuracy: %0.3f (+/-%0.03f) for params: %r" % (mean, std * 2, params))
    print()

    return classifier


def plot_correlation_matrix(lexicon_scores, encoded_labels, title="Correlation Matrix"):
    df = pd.DataFrame(lexicon_scores, columns=["gaming_score", "technology_score"])
    df["label"] = encoded_labels

    corr = df.corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


def generate_classification_report(y_pred, y_test, target_labels, title):
    print("Classification Report")

    print(classification_report(y_test, y_pred, target_names=target_labels))
    cm = confusion_matrix(y_test, y_pred, labels=target_labels)

    plot_confusion_matrix(cm, classes=target_labels, title=title)

    print("Final Accuracy")
    print(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
