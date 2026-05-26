# Mapping geo-acoustic distribution of global seafloor sediments via spatially constrained deep random forest
Luanxiao Zhao(School of Ocean and Earth Science, Tongji University) and Jianing Cai (School of Ocean and Earth Science, Tongji University)

1. The zip file provides latitude and longitude data for the global ocean division.
2. The BinnedData dataset contains labeled P-wave velocity data for seafloor sediments.
3. The “deepforest” folder provides the source code for the improved spatially constrained deep forest algorithm developed in this study.
Specifically, the traditional deep forest model is trained by setting is_spatial=False and calling fit_new(X_train, y_train). In contrast, the Spatially Constrained Deep Forest model is trained by setting is_spatial=True and calling fit(X_train, y_train, g_coords_train), where g_coords_train denotes the spatial coordinates of the training samples.
4. The Vp_10_fold_CV.py file provides the implementation of the traditional ten-fold cross-validation procedure.
5. The Vp_LOO_CV.py file implements the regional blind testing procedure. Specifically, the global ocean is partitioned into 77 labeled regions, with each iteration using data from 76 regions for model training and data from one withheld region for validation.
6. The Vp_Spatially_buffered_5_Flod_CV.py file provides the implementation of spatially buffered five-fold cross-validation. In this procedure, validation samples within a 1° radius of any training sample are excluded before model evaluation, ensuring spatial independence between the training and validation datasets.
7. The Vp_DF_DataBinning_Slices.py file is used to generate labels for model training.

**Note: This code requires NumPy 1.21.5 and sklearn 1.6.1.**
