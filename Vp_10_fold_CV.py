# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 17:14:31 2021

@author: Cloris
"""

import pandas as pd
import numpy as np
#from lce import LCERegressor
from keras.models import Model
from keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from deepforest import CascadeForestRegressor
from deepforest import cascade
from deepforest import _utils,_io
from deepforest._binner import Binner
from deepforest._layer import ClassificationCascadeLayer,CustomCascadeLayer,RegressionCascadeLayer
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score , mean_squared_error
from keras import initializers
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.utils import check_array, check_X_y
import warnings
warnings.filterwarnings('ignore')

class CascadeForestRegressor_gettree(cascade.CascadeForestRegressor):
    def __init__(self,
        n_bins=255,
        bin_subsample=200000,
        bin_type="percentile",
        max_layers=20,
        criterion="mse",
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        use_predictor=True,
        predictor="no_spatial",
        predictor_kwargs={},
        backend="custom",
        n_tolerant_rounds=2,
        delta=1e-5,
        partial_mode=False,
        n_jobs=None,
        random_state=None,
        verbose=1,
        sp_n_estimators=15,
        neighbors=450,
        is_spatial=True
    ):
        self.is_spatial=is_spatial
        if self.is_spatial:
            predictor = "spatial"
        else:
            predictor = "no_spatial"
        super().__init__(
            n_bins=n_bins,
            bin_subsample=bin_subsample,
            bin_type=bin_type,
            max_layers=max_layers,
            criterion=criterion,
            n_estimators=n_estimators,
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            use_predictor=use_predictor,
            predictor=predictor,
            predictor_kwargs=predictor_kwargs,
            backend=backend,
            n_tolerant_rounds=n_tolerant_rounds,
            delta=delta,
            partial_mode=partial_mode,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            sp_n_estimators=sp_n_estimators,
            neighbors=neighbors,
        )

    def predict(self, X,g_coords_test):
        X = check_array(X)

        if not self.is_fitted_:
            raise AttributeError("Please fit the model first.")
        self._check_input(X)

        if self.verbose > 0:
            print("{} Start to evalute the model:".format(_utils.ctime()))

        binner_ = self._get_binner(0)
        X_test = self._bin_data(binner_, X, is_training_data=False)
        X_middle_test_ = _utils.init_array(X_test, self.n_aug_features_)

        for layer_idx in range(self.n_layers_):
            layer = self._get_layer(layer_idx)

            if self.verbose > 0:
                msg = "{} Evaluating cascade layer = {:<2}"
                print(msg.format(_utils.ctime(), layer_idx))

            if layer_idx == 0:
                X_aug_test_ = layer.transform(X_test)
            elif layer_idx < self.n_layers_ - 1:
                binner_ = self._get_binner(layer_idx)
                X_aug_test_ = self._bin_data(
                    binner_, X_aug_test_, is_training_data=False
                )
                X_middle_test_ = _utils.merge_array(
                    X_middle_test_, X_aug_test_, self.n_features_
                )
                X_aug_test_ = layer.transform(X_middle_test_)
            else:
                binner_ = self._get_binner(layer_idx)
                X_aug_test_ = self._bin_data(
                    binner_, X_aug_test_, is_training_data=False
                )
                X_middle_test_ = _utils.merge_array(
                    X_middle_test_, X_aug_test_, self.n_features_
                )

                # Skip calling the `transform` if not using the predictor
                if self.use_predictor:
                    X_aug_test_ = layer.transform(X_middle_test_)

        if self.use_predictor:

            if self.verbose > 0:
                print("{} Evaluating the predictor".format(_utils.ctime()))
            if self.is_spatial:
                binner_ = self._get_binner(self.n_layers_)
                X_aug_test_ = self._bin_data(
                    binner_, X_aug_test_, is_training_data=False
                )

                X_middle_test_ = _utils.merge_array(
                    X_middle_test_, X_aug_test_, self.n_features_
                )

                predictor = self.buffer_.load_predictor(self.predictor_)
                _y , std = predictor.predict(X_middle_test_,g_coords_test)
            else:
                binner_ = self._get_binner(self.n_layers_)
                X_aug_test_ = self._bin_data(
                    binner_, X_aug_test_, is_training_data=False
                )

                X_middle_test_ = _utils.merge_array(
                    X_middle_test_, X_aug_test_, self.n_features_
                )

                predictor = self.buffer_.load_predictor(self.predictor_)
                _y = predictor.predict(X_middle_test_)
        else:
            if self.n_layers_ > 1:

                _y = layer.predict_full(X_middle_test_)
                temp=_y
                _y = _utils.merge_proba(_y, self.n_outputs_)
            else:
                # Directly merge results with one cascade layer only
                a=X_aug_test_
                _y = _utils.merge_proba(X_aug_test_, self.n_outputs_)

        return _y
np.random.seed(0)#for reproducibility  7
my_seed =0
for dl in range(1):
    matcontent = sio.loadmat('./BinnedData/LabeledData_add.mat')

    X_all = matcontent['X_all']
    BinnedVp = matcontent['BinnedVp'].flatten()
    BinnedLon = matcontent['BinnedLon'].flatten()
    BinnedLat = matcontent['BinnedLat'].flatten()
    X_all = X_all[:, 0:20]

    BinnedVp = BinnedVp.flatten()

    BinnedLon[BinnedLon < 0] = BinnedLon[BinnedLon < 0] + 360
    for id in range(1,2):
        kf = KFold(n_splits=10, shuffle=True, random_state=id)
        Preds = np.full(len(BinnedVp), np.nan)
        Preds_std = np.full(len(BinnedVp), np.nan)
        Preds_Tree = np.full((len(BinnedVp),16), np.nan)
        Feature=np.full((len(BinnedVp),16+31), np.nan)
        TrainTestIndex = []
        for train_index, test_index in kf.split(X_all):
            TrainTestIndex.append((train_index, test_index))

        for train_index, test_index in TrainTestIndex:
            train_index = train_index.flatten()
            test_index = test_index.flatten()
            X_train, X_test=X_all[train_index],X_all[test_index]
            y_train, y_test = BinnedVp[train_index],BinnedVp[test_index]
            lat_train, lat_test = BinnedLat.reshape(-1, 1)[train_index], BinnedLat.reshape(-1, 1)[test_index]
            lon_train, lon_test = BinnedLon.reshape(-1, 1)[train_index], BinnedLon.reshape(-1, 1)[test_index]
            lat_train = lat_train.flatten()
            lat_test = lat_test.flatten()
            lon_train = lon_train.flatten()
            lon_test = lon_test.flatten()
            g_coords_train = list(zip(lon_train, lat_train))
            g_coords_test = list(zip(lon_test, lat_test))
            #DeepForest = CascadeForestRegressor_gettree(is_spatial=False,sp_n_estimators=15,neighbors=450,max_layers=3,
            #                                           n_estimators=8, n_trees=80,random_state=1, n_jobs=3, verbose=0)
            #DeepForest.fit_new(X_train, y_train)

            DeepForest = CascadeForestRegressor_gettree(is_spatial=True, sp_n_estimators=15, neighbors=450,max_layers=3,
                                                        n_estimators=8, n_trees=80, random_state=1, n_jobs=3, verbose=0)
            DeepForest.fit(X_train, y_train,g_coords_train)
            y_pred= DeepForest.predict(X_test,g_coords_test)
            Preds[test_index] = y_pred.flatten()
        r2 = r2_score(BinnedVp, Preds)
        print(r2)



