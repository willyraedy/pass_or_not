import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def make_confusion_matrix(X_test, y_test, model, labels, title, threshold=0.5):
    # Predict class 1 if probability of being in class 1 is greater than threshold
    # (model.predict(X_test) does this automatically with a threshold of 0.5)
    y_predict = (model.predict_proba(X_test)[:, 1] >= threshold)
    fraud_confusion = confusion_matrix(y_test, y_predict)
    plt.figure(dpi=80)
    sns.heatmap(fraud_confusion, cmap='Blues', annot=True, square=True, fmt='d',
           xticklabels=labels,
           yticklabels=labels)
    plt.title(title)
    plt.xlabel('prediction')
    plt.ylabel('actual')

def report_single_model_metrics(scores, metrics=['roc_auc', 'accuracy', 'precision']):
    rows = []
    for metric in metrics:
        rows.append(dict(
            metric=metric,
            mean=scores['test_' + metric].mean(),
            std=scores['test_' + metric].std(),
            in_sample=scores['train_' + metric].mean()
        ))
    return pd.DataFrame(rows)


def report_grid_results(X, y, estimator, param_grid, scoring='roc_auc', cv=5, return_train_score=True):
    gr = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        return_train_score=return_train_score
    )
    gr.fit(X, y)
    df = pd.DataFrame(gr.cv_results_)
    results = df[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'rank_test_score']].sort_values('rank_test_score')
    return results, gr.best_params_

def run_pipeline(
    raw_data,
    features,
    estimator,
    param_grid
):
    # get matching train/test split with new feature set
    X, y = raw_data[features], raw_data['third_reading']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99, stratify=y)
    # optimize parameters
    _, best_params = report_grid_results(
        X_train,
        y_train,
        estimator(),
        param_grid=param_grid
    )
    # score optimal model
    scores_tuned = cross_validate(
        estimator(**best_params),
        X_train,
        y_train,
        return_train_score=True,
        scoring=['roc_auc', 'accuracy', 'precision'],
        cv=5
    )
    print('Best params:', best_params)
    # return formatted results
    return report_single_model_metrics(scores_tuned)
