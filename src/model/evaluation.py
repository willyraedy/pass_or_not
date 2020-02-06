import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

all_proba_metrics = ['roc_auc']
all_non_proba_metrics = ['accuracy', 'precision']

def cross_validate(
  estimator,
  X,
  y,
  scoring: list,
  fit_params={},
  cv=5,
  standardize=False,
  random_seed=42,
  stratify_on_target=True
):
    """
    Implements K Fold cross validation, returns metrics and estimators


    """
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_seed)
    proba_metrics = [x for x in scoring if x in all_proba_metrics]
    non_proba_metrics = [x for x in scoring if x in all_non_proba_metrics]

    X_train_np = np.array(X)
    y_train_np = np.array(y)

    results = defaultdict(list)
    for train_ind, val_ind in kf.split(X_train_np, y_train_np):

        X_tr, y_tr = X_train_np[train_ind], y_train_np[train_ind]
        X_val, y_val = X_train_np[val_ind], y_train_np[val_ind]

        if standardize:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)

        lm = estimator(**fit_params)
        fit = lm.fit(X_tr, y_tr)

        in_sample_preds = fit.predict(X_tr)
        out_sample_preds = fit.predict(X_val)
        for metric in non_proba_metrics:
          score_fn = getattr(metrics, metric + '_score')
          results['test_' + metric].append(score_fn(y_true=y_val, y_pred=out_sample_preds))
          results['train_' + metric].append(score_fn(y_true=y_tr, y_pred=in_sample_preds))

        if getattr(lm, 'predict_proba', None) and callable(lm.predict_proba):
          in_sample_preds = fit.predict_proba(X_tr)
          out_sample_preds = fit.predict_proba(X_val)
          for metric in proba_metrics:
            score_fn = getattr(metrics, metric + '_score')
            results['test_' + metric].append(score_fn(y_true=y_val, y_pred=out_sample_preds))
            results['train_' + metric].append(score_fn(y_true=y_tr, y_pred=in_sample_preds))

        results['estimators'].append(lm)

    return results

def make_confusion_matrix(X_test, y_test, model, labels, title, threshold=0.5):
    y_predict = (model.predict_proba(X_test)[:, 1] >= threshold)
    cmx = metrics.confusion_matrix(y_test, y_predict)
    plt.figure(dpi=80)
    sns.heatmap(cmx, cmap='Blues', annot=True, square=True, fmt='d',
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
            mean=np.array(scores['test_' + metric]).mean(),
            std=np.array(scores['test_' + metric]).std(),
            in_sample=np.array(scores['train_' + metric]).mean()
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
    results = df[[
      'params',
      'mean_test_score',
      'std_test_score',
      'mean_train_score',
      'rank_test_score'
    ]].sort_values('rank_test_score')
    return results, gr.best_params_

def run_pipeline(
    raw_data,
    features,
    estimator,
    param_grid,
    standardize=False
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
        estimator,
        X_train,
        y_train,
        fit_params=best_params,
        scoring=['roc_auc', 'accuracy', 'precision'],
        cv=5,
        standardize=standardize
    )
    print('Best params:', best_params)
    # return formatted results
    return report_single_model_metrics(scores_tuned)
