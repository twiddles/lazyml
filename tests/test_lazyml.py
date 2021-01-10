#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `lazyml` package."""

import pytest
import numpy as np
from lazyml.supervised import LazyClassifier
from lazyml.supervised import LazyRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import datasets


def test_classification():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.5, random_state=0)

    clf = LazyClassifier()
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    print(models)


def test_regression():
    boston = datasets.load_boston()
    X, y = boston.data, boston.target
    X = X.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.5, random_state=0)

    reg = LazyRegressor()
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    print(models)
