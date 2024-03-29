#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import random

# Configure if you want plots/prints
from sklearn.tree import DecisionTreeClassifier

verbose = True

# loading in our data
df = pd.read_csv("pd_speech_features.csv")

df['Parkinsons'] = 'Yes'
df.loc[df['class'] == 0, 'Parkinsons'] = 'No'
max_len = sum(df['class'] == 1) / len(df.index)
random.seed(0)

# Splitting data to X and y

X, y = df.iloc[:, :-2].values, df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

stdsc = StandardScaler()
X_train = stdsc.fit_transform(X_train)
X_test = stdsc.transform(X_test)
final_models = []


def df_summary():
    print(df.head())
    df['Parkinsons'].value_counts().plot.pie(autopct='%.1f%%')
    plt.show()


def logistic():
    lr_base = LogisticRegressionCV(random_state=0, max_iter=10000)
    lr_base.fit(X_train, y_train)
    if verbose:
        print('LR Base training accuracy:', lr_base.score(X_train, y_train))
        print('LR Base Test accuracy:', lr_base.score(X_test, y_test))
    # Add to our final models to compare
    final_models.append(('LR base', lr_base, 'LR'))


logistic()


def svc():
    svc_base = SVC(random_state=0)
    svc_base.fit(X_train, y_train)
    if verbose:
        print('SVC Base Training accuracy:', svc_base.score(X_train, y_train))
        print('SVC Base Test accuracy:', svc_base.score(X_test, y_test))
    # Add to our final models to compare
    final_models.append(('SVC base', svc_base, 'SVC'))


svc()


# Evaluating basic models

# This model is performing reasonably well, Maybe reduce the number of features

def randomforest():
    feat_labels = df.columns[1:]
    forest = RandomForestClassifier(n_estimators=500,
                                    random_state=1)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    sfm = SelectFromModel(forest, prefit=True, max_features=20)
    X_sel = sfm.transform(X_train)
    X_sel_test = sfm.transform(X_test)

    if verbose:
        print('Number of features',
              X_sel.shape[1])

        for f in range(X_sel.shape[1]):
            print("%2d) %-*s %f" % (f + 1, 30,
                                    feat_labels[indices[f]],
                                    importances[indices[f]]))
        plt.title('Feature Importance')
        plt.bar(range(X_sel.shape[1]),
                importances[indices][:20],
                align='center')

        plt.xticks(range(X_sel.shape[1]),
                   feat_labels[indices][:20], rotation=90)
        plt.xlim([-1, X_sel.shape[1]])
        plt.tight_layout()
        # plt.savefig('images/04_09.png', dpi=300)
        plt.show()
    return [X_sel, X_sel_test]


# Refitting the model
randomforest()


def k_neighbors():
    pipe_svc = make_pipeline(KNeighborsClassifier(n_jobs=-1))

    param_range = [1, 2, 5, 10, 25, 50]
    leaf_size = [10, 20, 30, 40, 50, 60, 100]
    param_grid = [{'kneighborsclassifier__n_neighbors': param_range,
                   'kneighborsclassifier__weights': ['uniform', 'distance'],
                   'kneighborsclassifier__p': [1, 2],
                   'kneighborsclassifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                   'kneighborsclassifier__leaf_size': leaf_size}]
    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring='balanced_accuracy',
                      refit=True,
                      cv=10,
                      n_jobs=-1)
    gs_KN = gs.fit(X_train, y_train)
    if verbose:
        print(f'KNeighbors Best score: {gs_KN.best_score_}')
        print(f'KNeighbors best params: {gs_KN.best_params_}')

    kn_class = gs_KN.best_estimator_

    # note that we do not need to refit the classifier
    # because this is done automatically via refit=True.

    if verbose:
        print('KNeighbors Test accuracy: %.3f' % kn_class.score(X_test, y_test))
    # Add to our final models to compare
    final_models.append(('KNeighbors', kn_class, 'KN'))


k_neighbors()


def plot_scaled(train_scores, test_scores, param_range):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean,
             color='blue', marker='o',
             markersize=5, label='Training accuracy')

    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')

    plt.plot(param_range, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='Validation accuracy')

    plt.fill_between(param_range,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1.0])
    plt.tight_layout()
    # plt.savefig('images/06_06.png', dpi=300)
    plt.show()


def scaled():
    pipe_svc_pca = make_pipeline(PCA(0.975),
                                 SVC(random_state=0))

    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100]
    train_scores, test_scores = validation_curve(
        estimator=pipe_svc_pca,
        X=X_train,
        y=y_train,
        param_name='svc__C',
        param_range=param_range,
        cv=10)

    if verbose:
        plot_scaled(train_scores, test_scores, param_range)


# scaled()


def grid_search():
    pipe_svc = make_pipeline(PCA(0.975),
                             SVC(max_iter=-1, random_state=0))

    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100, 1000]
    param_grid = [{'svc__C': param_range,
                   'svc__kernel': ['linear']},
                  {'svc__C': param_range,
                   'svc__gamma': param_range,
                   'svc__kernel': ['rbf']}]
    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring='balanced_accuracy',
                      refit=True,
                      cv=10,
                      n_jobs=-1)
    gs_PCA = gs.fit(X_train, y_train)
    if verbose:
        print(f'PCA.95 SVC Best score: {gs_PCA.best_score_}')
        print(f'PCA.95 SVC best params: {gs_PCA.best_params_}')

    clf_pca_95 = gs_PCA.best_estimator_

    # note that we do not need to refit the classifier
    # because this is done automatically via refit=True.

    if verbose:
        print('PCA SVD Test accuracy: %.3f' % clf_pca_95.score(X_test, y_test))
    # Add to our final models to compare
    final_models.append(('SVC with PCA of 0.975', clf_pca_95, 'SVC'))
    pipe_LR_PCA95 = make_pipeline(PCA(0.975),
                                  LogisticRegressionCV(random_state=0, max_iter=10000))

    gs = pipe_LR_PCA95
    clf_pca_95LR = gs.fit(X_train, y_train)

    # note that we do not need to refit the classifier
    # because this is done automatically via refit=True.

    if verbose:
        print('PCA Logistic Test accuracy: %.3f' % clf_pca_95LR.score(X_test, y_test))
    # Add to our final models to compare
    final_models.append(('LR with PCA of 0.975', clf_pca_95LR, 'LR'))


grid_search()


def pipe_randomforest():
    pipe_rf = make_pipeline(RandomForestClassifier(random_state=0))
    pipe_rf.fit(X_train, y_train)

    if verbose:
        print('Random Forest Train accuracy: %.3f' % pipe_rf.score(X_train, y_train))
        print('Random Forest Test accuracy: %.3f' % pipe_rf.score(X_test, y_test))
    # Add to our final models to compare
    final_models.append(('Random Forest', pipe_rf, 'RF'))


pipe_randomforest()


def decision_tree():
    pipe_svc = make_pipeline(DecisionTreeClassifier(random_state=0))

    samples_range = [2, 5, 10, 20, 50, 100]
    param_grid = [{'decisiontreeclassifier__criterion': ['gini', 'entropy'],
                   'decisiontreeclassifier__splitter': ['best', 'random'],
                   'decisiontreeclassifier__max_depth': samples_range,
                   'decisiontreeclassifier__min_samples_split': samples_range,
                   'decisiontreeclassifier__class_weight': [{0: 0.75, 1: 0.25},
                                                            {0: 0.25, 1: 0.75},
                                                            'balanced']}]
    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring='balanced_accuracy',
                      refit=True,
                      cv=10,
                      n_jobs=-1)
    gs_Decision = gs.fit(X_train, y_train)
    if verbose:
        print(f'DecisionTree Best score: {gs_Decision.best_score_}')
        print(f'DecisionTree best params: {gs_Decision.best_params_}')

    decision_class_s = gs_Decision.best_estimator_

    # note that we do not need to refit the classifier
    # because this is done automatically via refit=True.

    if verbose:
        print('DecisionTree Test accuracy: %.3f' % decision_class_s.score(X_test, y_test))
    # Add to our final models to compare
    final_models.append(('DecisionTree', decision_class_s, 'DT'))


decision_tree()


def compare():
    cv = StratifiedKFold(n_splits=3)

    all_results = []
    all_names = []
    dict_all = {}
    for var in range(len(final_models)):
        model_name = final_models[var][0]
        model = final_models[var][1]
        train = X_train
        cv_results = cross_val_score(model, X_test, y_test, cv=cv, scoring='balanced_accuracy')

        dict_all[model_name] = cv_results
        all_results.append(cv_results)
        all_names.append(model_name)
    df_models = pd.DataFrame.from_dict(dict_all, orient='index', columns=['one', 'two', 'three'])
    print(df_models)
    df_models.to_csv("all_models.csv")

    # fig.show()

    fig = plt.figure(figsize=(8, 4))
    fig.suptitle('All models compared')
    ax = fig.add_subplot(111)
    plt.boxplot(all_results)  # fill with color)
    ax.set_xticklabels(all_names, rotation=90)
    ax.set_ylim(0.50, 1)
    plt.plot(max_len)
    fig.show()

    plt.show()


compare()
