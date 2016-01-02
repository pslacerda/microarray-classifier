# coding: utf-8
"""
    Pedro Sousa Lacerda, 2015

"""

import glob
import pandas as pd
import numpy as np
import sys
import re
import collections
import itertools
import operator
import functools

from matplotlib import pyplot as plt
from matplotlib.pyplot import (subplot, plot, show, xticks, pcolor, colorbar,
                               legend, xlabel, ylabel, axvline, tight_layout,
                               bar, errorbar, ylim, xlim, savefig, text, gca)
from numpy import arange, abs, corrcoef, argsort, zeros

from sklearn import (preprocessing, decomposition, pipeline, grid_search,
                     feature_selection, lda, qda, svm)

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB

from sklearn.grid_search import GridSearchCV, ParameterGrid


def load_data(ifname, class_re):
    data = pd.read_table(ifname)

    data = data.drop(data.columns[[0, -1]], axis=1)
    data = data.drop(data.columns[1 + arange(len(data.columns))%2==1], axis=1)
    data = data.dropna(axis=0, how='all')
    data = data.transpose()

    labels = (class_re.search(l) for l in data.index)
    labels = [m.group() if m else '' for m in labels]
    label_encoder = preprocessing.LabelEncoder()

    X = data.values
    Y = label_encoder.fit_transform(labels)
    return X, Y


def compute_score(clf, X, Y):
    scores = cross_val_score(clf, X, Y, 'accuracy', NFOLDS, n_jobs=-1)
    return np.mean(scores), np.std(scores)


def make_grid(est, params):
    return grid_search.GridSearchCV(est, params, scoring='accuracy', cv=NFOLDS)

def errorbar(x, y, err, **kw):
    plt.plot(x, y, alpha=.8, lw=2, **kw)
    if 'label' in kw:
        del kw['label']
    plt.errorbar(x, y, yerr=err, alpha=.8, lw=.5, capsize=8, mew=2, **kw)

def figure0(X):
    plot(np.var(X, axis=0), lw=2, alpha=.8)
    xlabel(u"Gene")
    ylabel(u"Variância")

def figure1(classifiers, X, Y):
    Xscaled = preprocessing.scale(X)
    Xnormalized = preprocessing.normalize(X)

    raw = []
    scaled = []
    normalized = []

    for name, color, clf in classifiers:
        raw.append(compute_score(clf, X, Y))
        scaled.append(compute_score(clf, Xscaled, Y))
        normalized.append(compute_score(clf, Xnormalized, Y))

    raw = np.array(raw).T
    scaled = np.array(scaled).T
    normalized = np.array(normalized).T

    hpos = arange(len(classifiers))
    width = 0.3

    bar(hpos, raw[0], width, yerr=raw[1], color='r', alpha=.5)
    bar(hpos+width, scaled[0], width, yerr=scaled[1], color='g', alpha=.5)
    bar(hpos+width*2, normalized[0], width, yerr=normalized[1], color='b', alpha=.5)

    ylabel(u"Acurácia")
    xticks(hpos+width, [clf[0] for clf in classifiers])


def figure2(classifiers, X, Y, n_features_list):

    rf = RandomForestClassifier(random_state=1)
    rf.fit(X, Y)

    Xscaled = preprocessing.scale(X)
    inds = argsort(rf.feature_importances_)

    for name, color, clf in classifiers:
        scores = []
        for nfeatures in n_features_list:
            Xbest_inds = inds[::-1][:nfeatures]
            Xbest = Xscaled[:,Xbest_inds]
            scores.append(compute_score(clf, Xbest, Y))

        scores = np.array(scores).T
        label = u'%s\nacc %.2f' % (name, np.max(scores[0]))
        errorbar(n_features_list, scores[0], scores[1], label=label, color=color)

    legend(numpoints=1, ncol=len(classifiers), loc=4, fontsize=10)
    xlabel(u"Número de genes")
    ylabel(u"Acurácia")


def figure3(classifiers, X, Y, n_components_list, plot_n_components=9):

    Xscaled = preprocessing.scale(X)

    pca = decomposition.PCA(max(n_components_list))
    pca.fit(Xscaled)

    # subplot(211)
    plot(pca.explained_variance_, lw=2, alpha=.8)

    xlabel(u'Número de componentes')
    ylabel(u'Variância explicada')

    for name, color, clf in classifiers:
        pipe = pipeline.Pipeline([
            ('pca', pca),
            ('clf', clf)
        ])
        params = {'pca__n_components': n_components_list}
        grid = make_grid(pipe, params)
        grid.fit(Xscaled, Y)

        nc = grid.best_estimator_.named_steps['pca'].n_components
        label = u'%s\nacc   %.2f\nncomp %d' % (name, grid.best_score_, nc)
        axvline(nc, lw=3, ls=':', label=label, color=color)

    legend(numpoints=1, ncol=len(classifiers), loc=1, fontsize=10)

    # for i in range(plot_n_components):
    #     component = pca.components_[i]
    #     label = '#%d componente' % (i+1)
    #     plt.hist(component, histtype='step', alpha=.5, label=label, bins=100)
    # legend(loc=4, fontsize=10)


def figure4(classifiers, X, Y, n_features_list):

    Xscaled = preprocessing.scale(X)
    anova = feature_selection.SelectKBest(feature_selection.f_classif)

    # subplot(211)
    for name, color, clf in classifiers:
        pipe = pipeline.Pipeline([
            ('anova', anova),
            ('clf', clf)
        ])
        params = {'anova__k': n_features_list}
        grid = make_grid(pipe, params)
        grid.fit(Xscaled, Y)

        cv_scores = [s.cv_validation_scores for s in grid.grid_scores_]
        scores = np.mean(cv_scores, axis=1)
        stds = np.std(cv_scores, axis=1)

        label = u'%s\nacc %.2f' % (name, np.max(scores))
        errorbar(n_features_list, scores, stds, color=color, label=label)

    legend(numpoints=1, ncol=len(classifiers), loc=4, fontsize=10)
    xlabel(u"Número de genes")
    ylabel(u"Acurácia")

    # subplot(212)
    # for i in n_features_list:
    #     component = pca.components_[i]
    #     label = '#%d componente' % (i+1)
    #     plt.hist(component, histtype='step', alpha=.5, label=label, bins=100)
    # legend(loc=4, fontsize=10)


def figure5(classfiers, X, Y, n_features_list, n_components_list):


    class Figure5Grid(GridSearchCV):
        class ParameterGrid:
            def __init__(self, param_grid):
                self.anova__k = param_grid['anova__k']
                self.pca__n_components = param_grid['pca__n_components']

            def __iter__(self):
                for k in self.anova__k:
                    for n_components in self.pca__n_components:
                        if k < n_components:
                            yield {
                                'anova__k': k,
                                'pca__n_components': n_components,
                            }
            def __len__(self):
                return len(iter(self))
        def fit(self, X, y=None):
            return self._fit(X, y, self.ParameterGrid(self.param_grid))

    Xscaled = preprocessing.scale(X)
    pca = decomposition.PCA(max(n_components_list))
    anova = feature_selection.SelectKBest(feature_selection.f_classif)

    grids = {}
    for name, color, clf in classifiers:
        pipe = pipeline.Pipeline([
            ('pca', pca),
            ('anova', anova),
            ('clf', clf)
        ])
        params = {
            'pca__n_components': n_components_list,
            'anova__k': n_features_list
        }
        grid = Figure5Grid(pipe, params, scoring='accuracy', cv=NFOLDS)
        grid.fit(Xscaled, Y)

        grids[name] = grid

    # subplot(211)
    for name, color, clf in classifiers:
        grid = grids[name]
        nc = grid.best_params_['pca__n_components']

        scores = (s for s in grid.grid_scores_)
        scores = (s for s in scores if s.parameters['pca__n_components'] == nc)

        cv_scores = [s.cv_validation_scores for s in scores]

        scores = np.mean(cv_scores, axis=0)
        stds = np.std(cv_scores, axis=0)

        label = u'%s\nacc   %.2f\nncomp %d' % (name, np.max(scores), nc)
        errorbar(n_features_list, scores, stds, color=color, label=label)

    legend(numpoints=1, ncol=len(classifiers), loc=4, fontsize=10)
    gca().get_xaxis().set_visible(False)
    xlabel(u"Espaço de busca")
    ylabel(u"Acurácia")

    # subplot(212)
    # for name, color, clf in classifiers:
    #     grid = grids[name]
    #     k = grid.best_params_['anova__k']
    #
    #     pca.fit()
    #
    #
    #     pca = grid.best_estimator_



if __name__ == '__main__':
    REGEX = re.compile(sys.argv[1])
    INPUT = sys.argv[2]
    NFOLDS = 10

    classifiers = [
        ('SVC',   '#00995C', svm.SVC(kernel='linear', class_weight='auto', random_state=1)),
        ('LSVC',  '#5C991F', svm.LinearSVC(class_weight='auto',random_state=1)),
        ('QDA',   '#995C1F', qda.QDA()),
        ('LDA',   '#9966FF', lda.LDA()),
        ('RF',    '#991F5C', RandomForestClassifier(class_weight='auto', random_state=1)),
    ]

    X, Y = load_data(INPUT, REGEX)
    Xscaled = preprocessing.scale(X)

    N = X.shape[0]
    attr_range = range(1, N/2, 5)

    def save(fname):
        savefig(fname, bbox_inches='tight', transparence=True)

    plt.figure()
    figure0(X)
    save('figures/0.svg')

    # plt.figure()
    # figure1(classifiers, X, Y)
    # save('figures/1.svg')
    #
    # plt.figure()
    # ylim([.4, 1])
    # figure2(classifiers, X, Y, attr_range)
    # save('figures/2.svg')
    #
    # plt.figure()
    # xlim([0, attr_range[-1]+1])
    # figure3(classifiers, X, Y, attr_range)
    # save('figures/3.svg')
    #
    # plt.figure()
    # ylim([.4, 1])
    # xlim([0, N/2])
    # figure4(classifiers, X, Y, attr_range)
    # save('figures/4.svg')

    plt.figure()
    ylim([.4, 1])
    xlim([0, N/2])
    figure5(classifiers, X, Y, attr_range, attr_range)
    save('figures/5.svg')
