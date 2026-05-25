import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,VotingRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from xgboost import XGBRegressor
import os
#from tabpfn import TabPFNRegressor
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import warnings
warnings.filterwarnings('ignore')

def haversine_distance(coords_lonlat, core_lonlat, earth_radius=6371.0):
    coords_rad = np.radians(coords_lonlat)
    core_rad = np.radians(core_lonlat)
    lon1, lat1 = core_rad[0], core_rad[1]
    lon2 = coords_rad[:, 0]
    lat2 = coords_rad[:, 1]
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = earth_radius * c
    return distances
class SpatialRandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        sample_mode: str = "cluster",
        sample_by: str = "neighbors",
        neighbors: int = 500,
        max_distance: float = 150000,
        min_points_distance: int = 100,
        **decision_tree_args,
    ):
        self.estimators = []
        for i in range(n_estimators):

            et = ExtraTreesRegressor(n_estimators=80, n_jobs=-1,random_state=42)
            rf = RandomForestRegressor(n_estimators=80,n_jobs=-1,random_state=42)
            mini_ensemble = VotingRegressor(
                estimators=[('et', et),('rf', rf)],
                n_jobs=-1
            )

            self.estimators.append(mini_ensemble)
        if sample_by == "distance" and max_distance == 150000:
            warnings.warn(
                "It seems that you have selected the 'distance'-sampling mode,\
                     but the parameter max_distance is still the default. Make\
                     sure to adapt the max_distance parameter to your dataset."
            )
        self.n_estimators = n_estimators
        self.sample_mode = sample_mode
        self.sample_by = sample_by
        # only relevant if sample_by == distance
        self.max_distance = max_distance
        self.min_points_distance = min_points_distance
        # only relevant if sample_by == "neighbors"
        self.neighbors = neighbors
        # init core points
        self.estimator_core_points = []

    def _sample_core_points(self, coords):
        if self.sample_mode == "cluster":
            # cluster coordinates with kmeans use centers as core points
            kmeans = KMeans(self.n_estimators)
            kmeans.fit(coords)
            core_points = kmeans.cluster_centers_

        elif self.sample_mode == "random":
            # select random coordinates from the train data as core points
            core_points = coords[np.random.permutation(len(coords)
                                                       )[:self.n_estimators]]
        else:
            raise NotImplementedError(
                "sample mode must be one of cluster, random"
            )
        return core_points

    def _sample_point_clouds(self, coords):

        point_clouds = []
        for core_point in self.estimator_core_points:
            # Compute distance of the core point to all coordinates
            dist_to_others=haversine_distance(coords,core_point)
            #dist_to_others = np.sqrt(np.sum((coords - core_point)**2, axis=1))
            if self.sample_by == "neighbors":
                # add fixed number of closest samples
                point_clouds.append(
                    np.argsort(dist_to_others)[:self.neighbors]
                )
            elif self.sample_by == "distance":
                # filter by distance
                point_with_lower_dist = np.where(
                    dist_to_others < self.max_distance
                )[0]
                # only use point clouds that are large enough! --> cannot fit a
                # decision tree on 5 points
                if len(point_with_lower_dist) > self.min_points_distance:
                    point_clouds.append(point_with_lower_dist)
            else:
                raise NotImplementedError(
                    "sample mode must be one of 'neighbors', 'distance'!"
                )
        return point_clouds

    def fit(self, x_train, y_train, coords_train):

        # convert to arrays
        x_train, y_train, coords_train = (
            np.array(x_train),
            np.array(y_train),
            np.array(coords_train),
        )
        assert (
            len(coords_train.shape) == 2 and coords_train.shape[1] == 2
        ), "coords test must have len 2 in dimension 1"

        # sample core points
        self.estimator_core_points = self._sample_core_points(coords_train)
        # assign samples to their core points
        # (one sample can be in several point clouds!)
        point_clouds = self._sample_point_clouds(coords_train)
        if len(point_clouds) < self.n_estimators:
            warnings.warn(
                f"Some point clouds had less than {self.min_points_distance}\
                     points and are therefore ignored.\
                     Consider increasing the parameter 'max_distance' to\
           include more points (recommended), or decrease 'min_points_distance'"
            )
            # correct number of estimators
            self.n_estimators = len(point_clouds)
            self.estimators = self.estimators[:self.n_estimators]
        # correct core points: Use center of gravity of each point clouds
        self.estimator_core_points = np.array(
            [
                np.mean(coords_train[cloud_inds], axis=0)
                for cloud_inds in point_clouds
            ]
        )
        # fit each point cloud to an estimator
        for i, sample_inds in enumerate(point_clouds):
            x_train_subset = x_train[sample_inds]
            y_train_subset = y_train[sample_inds]
            self.estimators[i].fit(x_train_subset, y_train_subset)
    

    def predict(self, x_test, coords_test=None, weighted=True, closest=False):

        # convert to arrays
        x_test = np.array(x_test)
        if coords_test is not None:
            coords_test = np.array(coords_test)
        assert (coords_test is not None) or weighted == False, (
            "If weighted=True, then coords_test is required."
        )
        # predict output with each base estimator
        y_pred = np.zeros((len(x_test), self.n_estimators))
        for i, estimator in enumerate(self.estimators):
            y_pred[:, i] = estimator.predict(x_test)
        # If no spatial weighting: Simply return average of estimators
        if not weighted and not closest:
            return np.mean(y_pred, axis=1)
        # if weighted: check that test coords are alright
        coords_test = np.array(coords_test)
        assert (
            len(coords_test.shape) == 2 and coords_test.shape[1] == 2
        ), "coords test must have len 2 in dimension 1"
        # compute distance of test samples to all core points
        dist_to_core_points = np.array([
            haversine_distance(coords_test, core_point)
            for core_point in self.estimator_core_points
        ]).T
        if closest:
            use_tree = np.argmin(dist_to_core_points, axis=1)
            return y_pred[np.arange(len(y_pred)), use_tree]

        # turn into probabilies
        if np.any(dist_to_core_points == 0):
            # special if test sample is exactly equal to one of the core points
            weights = np.array(
                [0 if dist != 0 else 1 for dist in dist_to_core_points]
            )
        else:
            # normal situation: weight dependent on spatial distance
            weights = 1 / dist_to_core_points
            weights = weights / np.expand_dims(np.sum(weights, axis=1), 1)

        # prediction is weighted sum
        #y_pred = np.sum(y_pred * weights, axis=1)

        weighted_Mean = np.sum(y_pred * weights, axis=1)
        weighted_mean =weighted_Mean.reshape(-1,1)
        weighted_mean=np.tile(weighted_mean,weights.shape[1])
        squared_diff = (y_pred - weighted_mean) ** 2
        weighted_squared_diff = squared_diff * weights
        weighted_variance = np.sum(weighted_squared_diff, axis=1) / np.sum(weights, axis=1)
        weighted_std = np.sqrt(weighted_variance)

        return weighted_Mean,weighted_std

    def _sample_by_distance_old(
        coords_train, nr_clouds=20, radius=150000, min_points=400
    ):

        # make distance matrix n x n
        dist = np.zeros((len(coords_train), len(coords_train)))
        for i, coord1 in enumerate(coords_train):
            for j, coord2 in enumerate(coords_train[i:]):
                dist[i, j + i] = np.linalg.norm(coord1 - coord2)
        # mirror distance matrix
        dist = dist + dist.T
        # make point clouds
        point_clouds = []
        for core_ind in np.random.permutation(len(dist)):
            dist_to_others = dist[core_ind]
            inds = np.where(dist_to_others < radius)[0]
            if len(inds) > min_points:
                point_clouds.append(inds)
            #             print("Cloud for core ind", core_ind, "has members", len(inds))
            if len(point_clouds) > nr_clouds:
                break
        return point_clouds
