# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 17:14:31 2021

@author: Cloris
"""

import pandas as pd
import numpy as np
from deepforest import cascade
from deepforest import _utils,_io
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score , mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.utils import check_array
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
        n_trees=80,
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
import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from tqdm import tqdm
# ==========================================
# 0. seed
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(42)
# ==========================================
# 1. The continental margin area
# ==========================================
ggg_file = 'GL_ELEVATION_M_ASL_SRTM15+V2.5m.ggg'
nc_file = 'Dataset_S2.nc'
margin_set = set()

if os.path.exists(ggg_file) and os.path.exists(nc_file):
    ggg = np.fromfile(ggg_file, dtype=np.float32).reshape(2160, 4320)
    nc_obj = Dataset(nc_file)
    LonRange, LatRange = nc_obj.variables['lon'][:], nc_obj.variables['lat'][:]
    nc_obj.close()
    lon_grid, lat_grid = np.meshgrid(LonRange, LatRange)

    # 提取浅海区域 (-250 <= ggg < 0) 合并入 margin_set
    shallow_mask = (ggg >= -250) & (ggg < 0)
    shallow_lons, shallow_lats = lon_grid[shallow_mask], lat_grid[shallow_mask]
    for lon, lat in tqdm(zip(shallow_lons, shallow_lats), total=len(shallow_lons), desc="提取浅海数据"):
        if lat>-60:
            margin_set.add((round(lon, 4), round(lat, 4)))
else:
    print("error")

for id in range(0, 1):
    file_name = f"area_mergin/area_{id}.mat"
    if os.path.exists(file_name):
        matcontent = sio.loadmat(file_name)
        area_final_list = matcontent['area'].reshape(-1, 2) if matcontent['area'].ndim != 2 else np.array(
            [item for sublist in [item.tolist() for item in matcontent['area'].flatten()] for item in sublist])
        for lon, lat in area_final_list[:, :2]:
            if lat > -60:
                margin_set.add((round(lon, 4), round(lat, 4)))

# ==========================================
# 2. Longhurst area
# ==========================================
all_area_coords = []
all_area_labels = []

for id in tqdm(range(0, 54), desc=" "):
    file_name = f"area_biology/area_{id}.mat"
    if os.path.exists(file_name):
        matcontent = sio.loadmat(file_name)
        area = matcontent['area']

        if area.ndim == 2:
            area_list = [item.tolist() for item in area.flatten()]
            area_final_list = np.array([item for sublist in area_list for item in sublist])
        else:
            area_final_list = area.reshape(-1, 2)

        if area_final_list.ndim == 2 and area_final_list.shape[1] >= 2:
            lon_arr, lat_arr = area_final_list[:, 0], area_final_list[:, 1]

            in_margin = np.array([(round(x, 4), round(y, 4)) in margin_set for x, y in zip(lon_arr, lat_arr)])
            for i in range(len(lon_arr)):
                all_area_coords.append([lon_arr[i], lat_arr[i]])
                label = (id * 2 + 1) if in_margin[i] else (id * 2)
                all_area_labels.append(label)

all_area_coords = np.vstack(all_area_coords)
all_area_labels = np.array(all_area_labels)
tree_areas = cKDTree(all_area_coords)

# ==========================================
# 3. The continental margin area + Longhurst
# ==========================================
labeled_file = 'BinnedData/LabeledData_add.mat'
labeled_data = sio.loadmat(labeled_file)
b_lon = labeled_data['BinnedLon'].flatten()
b_lat = labeled_data['BinnedLat'].flatten()
y = labeled_data['BinnedVp'].flatten()
X = labeled_data['X_all']
b_coords = np.column_stack((b_lon, b_lat))


_, indices = tree_areas.query(b_coords)
sample_regions = all_area_labels[indices]
# ==========================================
# 4. LOO-CV
# ==========================================
r2_results = {}
Preds = np.full(len(y), np.nan)

for test_id in range(108):
    test_mask = (sample_regions == test_id)
    train_mask = ~test_mask
    X_test, y_test = X[test_mask], y[test_mask]
    X_train, y_train = X[train_mask], y[train_mask]
    if len(X_test) == 0:
        r2_results[test_id] = None
        continue
    if len(X_train) == 0:
        r2_results[test_id] = None
        continue
    g_coords_train = list(zip(b_lon[train_mask], b_lat[train_mask]))
    g_coords_test = list(zip(b_lon[test_mask], b_lat[test_mask]))

    rf = CascadeForestRegressor_gettree(is_spatial=True, sp_n_estimators=15, neighbors=500, max_layers=5,
                                        n_estimators=12, n_trees=80, random_state=1, n_jobs=-1, verbose=0)
    rf.fit(X_train, y_train, g_coords_train)


    y_pred = rf.predict(X_test, g_coords_test)
    Preds[test_mask] = y_pred.flatten()


valid_plot_mask = ~np.isnan(Preds)
y_p = y[valid_plot_mask]
Preds_p = Preds[valid_plot_mask]

if len(y_p) > 0:
    print(f" R²: {r2_score(y_p, Preds_p):.4f}")
