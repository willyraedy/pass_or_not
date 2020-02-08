import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from imblearn.over_sampling import RandomOverSampler

all_proba_metrics = ['roc_auc']
all_non_proba_metrics = ['accuracy', 'precision', 'recall']


def cross_validate(
  estimator,
  X,
  y,
  scoring: list,
  fit_params={},
  cv=5,
  standardize=False,
  stratify_on_target=True,
  oversample=False,
  show_conf_matrices=False,
  threshold=0.5
):
    """
    Implements K Fold cross validation, returns metrics and estimators


    """
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
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
        if oversample:
            ros = RandomOverSampler(random_state=0)
            X_tr, y_tr = ros.fit_sample(X_tr,y_tr)

        lm = estimator(**fit_params)
        fit = lm.fit(X_tr, y_tr)
        model_has_pred_proba = getattr(lm, 'predict_proba', None) and callable(lm.predict_proba)

        if threshold != 0.5:
          assert model_has_pred_proba, 'Model does not allow threshold tuning'
          apply_thresh = np.vectorize(lambda x: 1 if x >= threshold else 0)
          in_sample_preds = apply_thresh(fit.predict_proba(X_tr)[:, 1])
          out_sample_preds = apply_thresh(fit.predict_proba(X_val)[:, 1])
        else:
          in_sample_preds = fit.predict(X_tr)
          out_sample_preds = fit.predict(X_val)

        for metric in non_proba_metrics:
          score_fn = getattr(metrics, metric + '_score')
          results['test_' + metric].append(score_fn(y_true=y_val, y_pred=out_sample_preds))
          results['train_' + metric].append(score_fn(y_true=y_tr, y_pred=in_sample_preds))

        if show_conf_matrices:
          print_confusion_matrix(
            metrics.confusion_matrix(y_val, out_sample_preds),
            ['no_third', 'yes_third'],
            'Confusion Matrix'
          )

        if model_has_pred_proba:
          in_sample_preds = fit.predict_proba(X_tr)
          out_sample_preds = fit.predict_proba(X_val)

          for metric in proba_metrics:
            score_fn = getattr(metrics, metric + '_score')
            results['test_' + metric].append(score_fn(y_true=y_val, y_score=out_sample_preds[:, 1]))
            results['train_' + metric].append(score_fn(y_true=y_tr, y_score=in_sample_preds[:, 1]))
          results['y_proba_preds'].append(out_sample_preds)

        results['estimators'].append(lm)
        results['y_preds'].append(out_sample_preds)
        results['y_true'].append(y_val)
        results['X_test'].append(X_val)

    return results

def print_confusion_matrix(conf_mx, labels, title):
    plt.figure(dpi=80)
    sns.heatmap(conf_mx, cmap='Blues', annot=True, square=True, fmt='d',
           xticklabels=labels,
           yticklabels=labels)
    plt.title(title)
    plt.xlabel('prediction')
    plt.ylabel('actual')

def make_confusion_matrix(X_test, y_test, model, labels, title, threshold=0.5):
    y_predict = (model.predict_proba(X_test)[:, 1] >= threshold)
    cmx = metrics.confusion_matrix(y_test, y_predict)
    print(cmx, labels, title)

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


def report_grid_results(X, y, estimator, param_grid, scoring='roc_auc', return_train_score=True):
    gr = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
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
    standardize=False,
):
    # get matching train/test split with new feature set
    X, y = raw_data[features], raw_data['third_reading']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99, stratify=y)

    if standardize:
      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_train)
    # optimize parameters
    _, best_params = report_grid_results(
        X_train,
        y_train,
        estimator(),
        param_grid=param_grid,
    )
    # score optimal model
    scores_tuned = cross_validate(
        estimator,
        X_train,
        y_train,
        fit_params=best_params,
        scoring=['roc_auc', 'accuracy', 'precision'],
        standardize=False
    )
    print('Best params:', best_params)
    # return formatted results
    return report_single_model_metrics(scores_tuned)

def comparison_pipeline(X, y, models, metrics):
  results = []
  other_info = []
  for name, param_grid, estimator in models:
      # standardize if model requires it
      std = StandardScaler()
      X_tr = std.fit_transform(X) if name in ['knn', 'log', 'svm'] else X

      _, best_params = report_grid_results(
          X_tr,
          y,
          estimator(),
          param_grid=param_grid,
      )
      scores = cross_validate(
          estimator,
          X_tr,
          y,
          fit_params=best_params,
          scoring=metrics,
          standardize=False
      )
      other_info.append({'model': name, 'scores': scores, 'best_params': best_params })

      # format results into a single row
      res = report_single_model_metrics(scores)
      res = res[['mean']].transpose()
      res.columns = ['roc_auc', 'accuracy', 'precision']
      res.index = [name]

      results.append(res)
  return pd.concat(results), other_info
