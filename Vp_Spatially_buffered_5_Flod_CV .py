# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 17:14:31 2021

@author: Cloris
"""

from deepforest import cascade
from deepforest import _utils
import scipy.io as sio
from sklearn.utils import check_array
import warnings
warnings.filterwarnings('ignore')
import numpy as np
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
import os
import random
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42)
# Exploring the depth limit
for dl in range(1):
    labeled_file = 'BinnedData/LabeledData_add.mat'
    labeled_data = sio.loadmat(labeled_file)
    b_lon = labeled_data['BinnedLon'].flatten()
    b_lat = labeled_data['BinnedLat'].flatten()
    y = labeled_data['BinnedVp'].flatten()
    X = labeled_data['X_all'] [:, 0:20]
    b_coords = np.column_stack((b_lon, b_lat))
    b_lon[b_lon < 0] = b_lon[b_lon < 0] + 360
    # ==========================================
    from sklearn.model_selection import KFold
    from scipy.spatial import cKDTree
    from sklearn.metrics import r2_score
    import numpy as np
    n_splits = 5
    buffer_dist = 1
    for id in range(10):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=id)
        r2_results = []
        Preds = np.full(len(y), np.nan)
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            final_train_idx = train_idx
            tree_train = cKDTree(b_coords[final_train_idx])
            distances_to_train, _ = tree_train.query(b_coords[test_idx], k=1)
            valid_test_mask = distances_to_train > buffer_dist
            final_test_idx = test_idx[valid_test_mask]
            # ----------------------------------------------
            if len(final_test_idx) == 0:
                print(f"Fold {fold + 1:2d}: error")
                continue
            X_train, y_train = X[final_train_idx], y[final_train_idx]
            X_test, y_test = X[final_test_idx], y[final_test_idx]
            g_coords_train = list(zip(b_lon[final_train_idx], b_lat[final_train_idx]))
            g_coords_test = list(zip(b_lon[final_test_idx], b_lat[final_test_idx]))


            #Spatially Constrained Deep forest
            rf = CascadeForestRegressor_gettree(is_spatial=True, sp_n_estimators=15,neighbors=450, max_layers=3, n_estimators=8, n_trees=80,
                                                random_state=1, n_jobs=1, verbose=0)
            rf.fit(X_train, y_train, g_coords_train)


            #Deep forest
            # rf = CascadeForestRegressor_gettree(is_spatial=True, sp_n_estimators=15,neighbors=450, max_layers=3, n_estimators=8, n_trees=80,
            #                                       random_state=1, n_jobs=1, verbose=0)
            #rf.fit_new(X_train, y_train)



            y_pred = rf.predict(X_test, g_coords_test)
            r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan
            r2_results.append(r2)
            Preds[final_test_idx] = y_pred.flatten()
        valid_plot_mask = ~np.isnan(Preds)
        y_plot = y[valid_plot_mask]
        Preds_plot = Preds[valid_plot_mask]
        print(f"{r2_score(y_plot, Preds_plot)}")





