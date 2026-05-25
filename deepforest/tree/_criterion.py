# encoding: utf-8
# module deepforest.tree._criterion
# from E:\anaconda\lib\site-packages\deepforest\tree\_criterion.cp39-win_amd64.pyd
# by generator 1.147
# no doc

# imports
import builtins as __builtins__ # <module 'builtins' (built-in)>
import numpy as np # E:\anaconda\lib\site-packages\numpy\__init__.py

# functions

def __pyx_unpickle_Enum(*args, **kwargs): # real signature unknown
    pass

# classes

class Criterion(object):
    """
    Interface for impurity criteria.
    
        This object stores methods on how to calculate how good a split is using
        different metrics.
    """
    def __getstate__(self, *args, **kwargs): # real signature unknown
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce_cython__(self, *args, **kwargs): # real signature unknown
        pass

    def __setstate_cython__(self, *args, **kwargs): # real signature unknown
        pass

    def __setstate__(self, *args, **kwargs): # real signature unknown
        pass

    __pyx_vtable__ = None # (!) real value is '<capsule object NULL at 0x000002598E46A8A0>'


class ClassificationCriterion(Criterion):
    """ Abstract criterion for classification. """
    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs): # real signature unknown
        pass

    __pyx_vtable__ = None # (!) real value is '<capsule object NULL at 0x000002598E46A900>'


class Entropy(ClassificationCriterion):
    """
    Cross Entropy impurity criterion.
    
        This handles cases where the target is a classification taking values
        0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
        then let
    
            count_k = 1 / Nm \sum_{x_i in Rm} I(yi = k)
    
        be the proportion of class k observations in node m.
    
        The cross-entropy is then defined as
    
            cross-entropy = -\sum_{k=0}^{K-1} count_k log(count_k)
    """
    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    __pyx_vtable__ = None # (!) real value is '<capsule object NULL at 0x000002598E46A9C0>'


class RegressionCriterion(Criterion):
    """
    Abstract regression criterion.
    
        This handles cases where the target is a continuous value, and is
        evaluated by computing the variance of the target values left and right
        of the split point. The computation takes linear time with `n_samples`
        by using ::
    
            var = \sum_i^n (y_i - y_bar) ** 2
                = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
    """
    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs): # real signature unknown
        pass

    __pyx_vtable__ = None # (!) real value is '<capsule object NULL at 0x000002598E46A960>'


class MSE(RegressionCriterion):
    """
    Mean squared error impurity criterion.
    
            MSE = var_left + var_right
    """
    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    __pyx_vtable__ = None # (!) real value is '<capsule object NULL at 0x000002598E46AA80>'


class FriedmanMSE(MSE):
    """
    Mean squared error impurity criterion with improvement score by Friedman
    
        Uses the formula (35) in Friedman's original Gradient Boosting paper:
    
            diff = mean_left - mean_right
            improvement = n_left * n_right * diff^2 / (n_left + n_right)
    """
    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    __pyx_vtable__ = None # (!) real value is '<capsule object NULL at 0x000002598E46AB40>'


class Gini(ClassificationCriterion):
    """
    Gini Index impurity criterion.
    
        This handles cases where the target is a classification taking values
        0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
        then let
    
            count_k = 1/ Nm \sum_{x_i in Rm} I(yi = k)
    
        be the proportion of class k observations in node m.
    
        The Gini Index is then defined as:
    
            index = \sum_{k=0}^{K-1} count_k (1 - count_k)
                  = 1 - \sum_{k=0}^{K-1} count_k ** 2
    """
    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    __pyx_vtable__ = None # (!) real value is '<capsule object NULL at 0x000002598E46AA20>'


class MAE(RegressionCriterion):
    """
    Mean absolute error impurity criterion
    
           MAE = (1 / n)*(\sum_i |y_i - f_i|), where y_i is the true
           value and f_i is the predicted value.
    """
    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    __pyx_vtable__ = None # (!) real value is '<capsule object NULL at 0x000002598E46AAE0>'


# variables with complex values

__loader__ = None # (!) real value is '<_frozen_importlib_external.ExtensionFileLoader object at 0x000002598E46A490>'

__spec__ = None # (!) real value is "ModuleSpec(name='deepforest.tree._criterion', loader=<_frozen_importlib_external.ExtensionFileLoader object at 0x000002598E46A490>, origin='E:\\\\anaconda\\\\lib\\\\site-packages\\\\deepforest\\\\tree\\\\_criterion.cp39-win_amd64.pyd')"

__test__ = {}

