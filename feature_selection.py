#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 21:24:53 2018

@author: josharnold
"""

from sklearn.feature_selection import VarianceThreshold

class selector:
    def remove_low_variance(X):
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        return sel.fit_transform(X)
        