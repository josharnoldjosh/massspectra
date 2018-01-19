#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:46:45 2018

@author: josharnold
"""

class grids:
    def default_opt_act_kern():
        optimizers = ['SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam']
        activations = ['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']
        kernal_init = ['uniform','lecun_uniform','normal','orthogonal','zero','one','glorot_normal','glorot_uniform', 'he_normal', 'he_uniform']
        return optimizers, activations, kernal_init
    
    def default_layer_weights():
        return [900], [800], [900]
    
    def default_dropout():
        return [0.2], [0.2]
    
    def default_batch_epochs():
        return [20], [10]