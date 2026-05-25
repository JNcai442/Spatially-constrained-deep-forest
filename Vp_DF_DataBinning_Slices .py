# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 17:18:23 2021

@author: Cloris
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from netCDF4 import Dataset
import numpy as np
import scipy.io as sio
import xlrd
import pandas as pd
import netCDF4
from netCDF4 import Dataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import math
def convert_to_spherical_coordinates(longitude, latitude, r):
    # 将经纬度转换为弧度
    lon_rad = math.radians(longitude)
    lat_rad = math.radians(latitude)

    # 计算球坐标
    x = r * math.cos(lat_rad) * math.cos(lon_rad)
    y = r * math.cos(lat_rad) * math.sin(lon_rad)
    z = r * math.sin(lat_rad)

    return x, y, z
nc_obj = Dataset('Dataset_S2.nc')
LonRange = nc_obj.variables['lon'][:]
LatRange = nc_obj.variables['lat'][:]
Refer = nc_obj.variables['z'][:]
R = 6370.856
f1 = np.full(2160 * 4320, np.nan)
f1 = f1.reshape(2160, 4320)
f2 = np.full(2160 * 4320, np.nan)
f2 = f2.reshape(2160, 4320)
f52 = np.full(2160 * 4320, np.nan)
f52 = f52.reshape(2160, 4320)
for i in range(2160):
    for j in range(4320):
        longitude = LonRange[j]
        latitude = LatRange[i]
        f1[i, j], f2[i, j], f52[i, j] = convert_to_spherical_coordinates(longitude, latitude, R)
# 5-arc minute pitch
nc_obj = Dataset('Dataset_S2.nc')     
LonRange = nc_obj.variables['lon'][:]
LatRange = nc_obj.variables['lat'][:]

for dl in range(1):
    PWL_data = sio.loadmat('BinnedData/Data_1.mat')
    Lons = np.array(PWL_data['lon']).reshape(-1, 1)
    Lats = np.array(PWL_data['lat']).reshape(-1, 1)
    Vp = np.array(PWL_data['PWL']).reshape(-1, 1)
    ggg = np.fromfile('95_grids/GL_LONGITUDE.5m.ggg', dtype=np.float32)
    #f1 = ggg.reshape(2160, 4320)
    ggg = np.fromfile('95_grids/GL_LATITUDE_DD.5m.ggg', dtype=np.float32)
    #f2 = ggg.reshape(2160, 4320)
    ggg = np.fromfile('95_grids/GL_VOLCANO_GVP.r200km.wct.5m.ggg', dtype=np.float32)
    f3 = ggg.reshape(2160, 4320)
    ggg = np.fromfile('95_grids/GL_ELEVATION_M_ASL_SRTM15+V2.5m.ggg', dtype=np.float32)
    f4 = ggg.reshape(2160, 4320)
    ggg = np.fromfile('95_grids/SS_DENSITY_KGM-3_SACD_Aquarius_MISSION_MEANx.5m.ggg', dtype=np.float32)
    f5 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_DEEP_QUAKES_NCEDC.r100km.wct.5m.ggg', dtype=np.float32)
    f6 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_ACTIVE_VOLCANO_GVP.r200km.wct.5m.ggg', dtype=np.float32)
    f7 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_ACTIVE_VOLCANO_GVP.r1000km.wct.5m.ggg', dtype=np.float32)
    f8 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_RIVERMOUTH_TSS_TGYR-1_ORNL.5m.ggg', dtype=np.float32)
    f9 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SC_CRUST_AGE_MA_Muller.5m.ggg', dtype=np.float32)
    f10 = ggg.reshape(2160, 4320) 
    
    
    ggg = np.fromfile('95_grids/GL_VOLCANO_GVP.r100km.wct.5m.ggg', dtype=np.float32)
    f11 = ggg.reshape(2160, 4320)  
    
    ggg = np.fromfile('95_grids/SF_GRAINSIZE_D84_MM_NGDC.5m.ggg', dtype=np.float32)
    f12 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_ACTIVE_VOLCANO_GVP.r100km.wct.5m.ggg', dtype=np.float32)
    f13 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_DEEP_QUAKES_NCEDC.r500km.wct.5m.ggg', dtype=np.float32)
    f14 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SEA_SALINITY_PSU_DECADAL_MEAN_woa13v2x.5m.ggg', dtype=np.float32)
    f15 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SHALLOW_QUAKES_NCEDC.r10km.wct.5m.ggg', dtype=np.float32)
    f16 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_ACTIVE_VOLCANO_GVP.r500km.wct.5m.ggg', dtype=np.float32)
    f17 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SS_WAVE_PERIOD_S_2012_12_WAVEWATCH3x.5m.ggg', dtype=np.float32)
    f18 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SEAMOUNTS_KIM.r100km.wct.5m.ggg', dtype=np.float32)
    f19 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_ACTIVE_SEAMOUNTS_KIM.r10km.wct.5m.ggg', dtype=np.float32)
    f20 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_ACTIVE_SEAMOUNTS_KIM.r50km.wct.5m.ggg', dtype=np.float32)
    f21 = ggg.reshape(2160, 4320)  
    
    ggg = np.fromfile('95_grids/GL_VOLCANO_GVP.r500km.wct.5m.ggg', dtype=np.float32)
    f22 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_ACTIVE_VOLCANO_GVP.r50km.wct.5m.ggg', dtype=np.float32)
    f23 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SEAMOUNTS_KIM.r10km.wct.5m.ggg', dtype=np.float32)
    f24 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_VOLCANO_GVP.r50km.wct.5m.ggg', dtype=np.float32)
    f25 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_DEEP_QUAKES_NCEDC.r200km.wct.5m.ggg', dtype=np.float32)
    f26 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_CLAYFRACTION_FRAC_NGDC.5m.ggg', dtype=np.float32)
    f27 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_ACTIVE_SEAMOUNTS_KIM.r100km.wct.5m.ggg', dtype=np.float32)
    f28 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_DIST_TO_TRENCH_KM_PLATES.5m.ggg', dtype=np.float32)
    f29 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_RIVERMOUTH_POC_TGCYR-1_ORNL.5m.ggg', dtype=np.float32)
    f30 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_ACTIVE_SEAMOUNTS_KIM.r500km.wct.5m.ggg', dtype=np.float32)
    f31 = ggg.reshape(2160, 4320)  
    
    ggg = np.fromfile('95_grids/SF_CALSIL_FRAC_DUTKIEWICZ.5m.ggg', dtype=np.float32)
    f32 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SC_CRUST_THICK_M_CRUST1s.5m.ggg', dtype=np.float32)
    f33 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_RIVERMOUTH_HCO3_TGCYR-1_ORNL.5m.ggg', dtype=np.float32)
    f34 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SEA_NITRATE_MCML_DECADAL_MEAN_woa13v2x.5m.ggg', dtype=np.float32)
    f35 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_DEEP_QUAKES_NCEDC.r1000km.wct.5m.ggg', dtype=np.float32)
    f36 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_GRAINSIZE_D16_MM_NGDC.5m.ggg', dtype=np.float32)
    f37 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_DEEP_QUAKES_NCEDC.r10km.wct.5m.ggg', dtype=np.float32)
    f38 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SL_GEOID_M_ABOVE_WGS84_NGA_egm2008.5m.ggg', dtype=np.float32)
    f39 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SS_BIOMASS_INVERTEBRATE_LOG10_MGCM2_WEI2010x.5m.ggg', dtype=np.float32)
    f40 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_DEEP_QUAKES_NCEDC.r50km.wct.5m.ggg', dtype=np.float32)
    f41 = ggg.reshape(2160, 4320)  
    
    ggg = np.fromfile('95_grids/SS_PHOTO_AVAIL_RAD_EINSTEIN_M-2_DAY_SNPP_VIIRS_MISSION_MEANx.5m.ggg', dtype=np.float32)
    f42 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_TERBIO_FRAC_DUTKIEWICZ.5m.ggg', dtype=np.float32)
    f43 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_RIVERMOUTH_DOC_TGCYR-1_ORNL.5m.ggg', dtype=np.float32)
    f44 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SC_CRUST_VP_MS_CRUST1s.5m.ggg', dtype=np.float32)
    f45 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SS_WAVE_HEIGHT_M_2012_12_WAVEWATCH3x.5m.ggg', dtype=np.float32)
    f46 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_TOC_PDW.5m.ggg', dtype=np.float32)
    f47 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SEA_SEA_OXYGEN_UTILIZATION_MOLM3_DECADAL_MEAN_woa13v2x.5m.ggg', dtype=np.float32)
    f48 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_RIVERMOUTH_CO2_TGCYR-1_ORNL.5m.ggg', dtype=np.float32)
    f49 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_TOT_SED_THICK_M_CRUST1_NOAA.5m.ggg', dtype=np.float32)
    f50 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SHALLOW_QUAKES_NCEDC.r1000km.wct.5m.ggg', dtype=np.float32)
    f51 = ggg.reshape(2160, 4320)  
    
    ggg = np.fromfile('95_grids/GL_LATITUDE_DD.5m.ggg', dtype=np.float32)
    #f52 = ggg.reshape(2160, 4320)
    
    ggg = np.fromfile('95_grids/SC_CRUST_DEN_KGM3_CRUST1s.5m.ggg', dtype=np.float32)
    f53 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_COSLATITUDE.5m.ggg', dtype=np.float32)
    #f54 = ggg.reshape(2160, 4320)
    
    ggg = np.fromfile('95_grids/GL_VOLCANO_GVP.r1000km.wct.5m.ggg', dtype=np.float32)
    f55 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SEA_PHOSPHATE_MCML_DECADAL_MEAN_woa13v2x.5m.ggg', dtype=np.float32)
    f56 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_DIST_TO_PLATE_BOUNDARY_KM_PLATES.5m.ggg', dtype=np.float32)
    f57 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SHALLOW_QUAKES_NCEDC.r50km.wct.5m.ggg', dtype=np.float32)
    f58 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SS_PIC_LOG_MOL_M3-1_MODIS_Aqua_MISSION_MEANx.5m.ggg', dtype=np.float32)
    f59 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SEAMOUNTS_KIM.r50km.wct.5m.ggg', dtype=np.float32)
    f60 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_ACTIVE_SEAMOUNTS_KIM.r200km.wct.5m.ggg', dtype=np.float32)
    f61 = ggg.reshape(2160, 4320)  
    
    ggg = np.fromfile('95_grids/SF_SHALLOW_QUAKES_NCEDC.r200km.wct.5m.ggg', dtype=np.float32)
    f62 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_TOT_SED_THICK_M_GLOBSED_Straume.5m.ggg', dtype=np.float32)
    f63 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SEA_TEMPERATURE_C_DECADAL_MEAN_woa13v2x.5m.ggg', dtype=np.float32)
    f64 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SS_BIOMASS_TOTAL_LOG10_MGCM2_WEI2010x.5m.ggg', dtype=np.float32)
    f65 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SEA_OXYGEN_MLL_DECADAL_MEAN_woa13v2x.5m.ggg', dtype=np.float32)
    f66 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SEA_SILICATE_MCML_DECADAL_MEAN_woa13v2x.5m.ggg', dtype=np.float32)
    f67 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SEAMOUNTS_KIM.r200km.wct.5m.ggg', dtype=np.float32)
    f68 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/CM_MANTLE_DEN_KGM3_CRUST1s.5m.ggg', dtype=np.float32)
    f69 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SS_POC_LOG_MOL_M3-1_MODIS_Aqua_MISSION_MEANx.5m.ggg', dtype=np.float32)
    f70 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SS_BIOMASS_MACROFAUNA_LOG10_MGCM2_WEI2010x.5m.ggg', dtype=np.float32)
    f71 = ggg.reshape(2160, 4320)  
    
    ggg = np.fromfile('95_grids/GL_DIST_TO_TRANSFORM_KM_PLATES.5m.ggg', dtype=np.float32)
    f72 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SHALLOW_QUAKES_NCEDC.r100km.wct.5m.ggg', dtype=np.float32)
    f73 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SEAMOUNTS_KIM.r500km.wct.5m.ggg', dtype=np.float32)
    f74 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_ACTIVE_SEAMOUNTS_KIM.r1000km.wct.5m.ggg', dtype=np.float32)
    f75 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SHALLOW_QUAKES_NCEDC.r500km.wct.5m.ggg', dtype=np.float32)
    f76 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_CURRENT_EAST_MS_2012_12_HYCOMx.5m.ggg', dtype=np.float32)
    f77 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SEA_OXYGEN_PCTSAT_DECADAL_MEAN_woa13v2x.5m.ggg', dtype=np.float32)
    f78 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_GRAINSIZE_D50_MM_NGDC.5m.ggg', dtype=np.float32)
    f79 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SS_WAVE_DIRECTION_DEG_2012_12_WAVEWATCH3x.5m.ggg', dtype=np.float32)
    f80 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_AVG_SEA_DENSITY_KGM3_DECADAL_MEAN_woa13x.5m.ggg', dtype=np.float32)
    f81 = ggg.reshape(2160, 4320)  
    
    ggg = np.fromfile('95_grids/SS_BIOMASS_MEIOFAUNA_LOG10_MGCM2_WEI2010x.5m.ggg', dtype=np.float32)
    f82 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SS_BIOMASS_MEIOFAUNA_LOG10_MGCM2_WEI2010x.5m.ggg', dtype=np.float32)
    f83 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SS_BIOMASS_FISH_LOG10_MGCM2_WEI2010x.5m.ggg', dtype=np.float32)
    f84 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SEA_CONDUCTIVITY_SM_DECADAL_MEAN_woa13v2x.5m.ggg', dtype=np.float32)
    f85 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_DIST_TO_COAST_KM_ETOPO.5m.ggg', dtype=np.float32)
    f86 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_CURRENT_NORTH_MS_2012_12_HYCOMx.5m.ggg', dtype=np.float32)
    f87 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SS_PHYTO_ABSORPTION_443NM_M-1_SNPP_VIIRS_MISSION_MEANx.5m.ggg', dtype=np.float32)
    f88 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_SEAMOUNTS_KIM.r1000km.wct.5m.ggg', dtype=np.float32)
    f89 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SS_CHLOROPHYLL_LOG_MG_M3_MODIS_Aqua_MISSION_MEANx.5m.ggg', dtype=np.float32)
    f90 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/GL_DIST_TO_RIDGE_KM_PLATES.5m.ggg', dtype=np.float32)
    f91 = ggg.reshape(2160, 4320)  
    
    ggg = np.fromfile('95_grids/SC_CRUST_VS_MS_CRUST1s.5m.ggg', dtype=np.float32)
    f92 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SS_WINDSPEED_MS-1_SACD_Aquarius_MISSION_MEANx.5m.ggg', dtype=np.float32)
    f93 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SS_BIOMASS_MEGAFAUNA_LOG10_MGCM2_WEI2010x.5m.ggg', dtype=np.float32)
    f94 = ggg.reshape(2160, 4320) 
    
    ggg = np.fromfile('95_grids/SF_AVG_SEA_SOUNDSPEED_MS_DECADAL_MEAN_woa13x.5m.ggg', dtype=np.float32)
    f95 = ggg.reshape(2160, 4320)
    f52 = f95


    f96 = sio.loadmat('data_literature/grl_porosity.mat')
    f96=f96['a']
    f96=f96.reshape(2160, 4320)


    f99=Dataset("G:/ML/Vp/data_literature/preduction/density.nc")
    f99=f99.variables['z'][:]
    f99=f99.reshape(2160, 4320)
    f54 = f99


    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    Scaler = MinMaxScaler()
    f1 = SI.fit_transform(f1.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f2 = SI.fit_transform(f2.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f3 = SI.fit_transform(f3.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f4 = SI.fit_transform(f4.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f5 = SI.fit_transform(f5.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f6 = SI.fit_transform(f6.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f7 = SI.fit_transform(f7.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f8 = SI.fit_transform(f8.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f9 = SI.fit_transform(f9.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f10 = SI.fit_transform(f10.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f11 = SI.fit_transform(f11.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f12 = SI.fit_transform(f12.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f13 = SI.fit_transform(f13.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f14 = SI.fit_transform(f14.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f15 = SI.fit_transform(f15.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f16 = SI.fit_transform(f16.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f17 = SI.fit_transform(f17.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f18 = SI.fit_transform(f18.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f19 = SI.fit_transform(f19.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f20 = SI.fit_transform(f20.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f21 = SI.fit_transform(f21.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f22 = SI.fit_transform(f22.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f23 = SI.fit_transform(f23.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f24 = SI.fit_transform(f24.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f25 = SI.fit_transform(f25.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f26 = SI.fit_transform(f26.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f27 = SI.fit_transform(f27.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f28 = SI.fit_transform(f28.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f29 = SI.fit_transform(f29.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f30 = SI.fit_transform(f30.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f31 = SI.fit_transform(f31.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f32 = SI.fit_transform(f32.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f33 = SI.fit_transform(f33.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f34 = SI.fit_transform(f34.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f35 = SI.fit_transform(f35.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f36 = SI.fit_transform(f36.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f37 = SI.fit_transform(f37.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f38 = SI.fit_transform(f38.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f39 = SI.fit_transform(f39.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f40 = SI.fit_transform(f40.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f41 = SI.fit_transform(f41.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f42 = SI.fit_transform(f42.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f43 = SI.fit_transform(f43.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f44 = SI.fit_transform(f44.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f45 = SI.fit_transform(f45.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f46 = SI.fit_transform(f46.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f47 = SI.fit_transform(f47.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f48 = SI.fit_transform(f48.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f49 = SI.fit_transform(f49.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f50 = SI.fit_transform(f50.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f51 = SI.fit_transform(f51.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f52 = SI.fit_transform(f52.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f53 = SI.fit_transform(f53.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f54 = SI.fit_transform(f54.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f55 = SI.fit_transform(f55.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f56 = SI.fit_transform(f56.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f57 = SI.fit_transform(f57.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f58 = SI.fit_transform(f58.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f59 = SI.fit_transform(f59.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f60 = SI.fit_transform(f60.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f61 = SI.fit_transform(f61.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f62 = SI.fit_transform(f62.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f63 = SI.fit_transform(f63.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f64 = SI.fit_transform(f64.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f65 = SI.fit_transform(f65.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f66 = SI.fit_transform(f66.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f67 = SI.fit_transform(f67.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f68 = SI.fit_transform(f68.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f69 = SI.fit_transform(f69.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f70 = SI.fit_transform(f70.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f71 = SI.fit_transform(f71.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f72 = SI.fit_transform(f72.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f73 = SI.fit_transform(f73.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f74 = SI.fit_transform(f74.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f75 = SI.fit_transform(f75.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f76 = SI.fit_transform(f76.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f77 = SI.fit_transform(f77.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f78 = SI.fit_transform(f78.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f79 = SI.fit_transform(f79.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f80 = SI.fit_transform(f80.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f81 = SI.fit_transform(f81.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f82 = SI.fit_transform(f82.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f83 = SI.fit_transform(f83.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f84 = SI.fit_transform(f84.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f85 = SI.fit_transform(f85.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f86 = SI.fit_transform(f86.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f87 = SI.fit_transform(f87.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f88 = SI.fit_transform(f88.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f89 = SI.fit_transform(f89.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f90 = SI.fit_transform(f90.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f91 = SI.fit_transform(f91.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f92 = SI.fit_transform(f92.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f93 = SI.fit_transform(f93.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f94 = SI.fit_transform(f94.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f95 = SI.fit_transform(f95.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f96 = SI.fit_transform(f96.reshape(-1, 1))
    SI = SimpleImputer(missing_values=np.nan, strategy='mean')
    f99 = SI.fit_transform(f99.reshape(-1, 1))

    scaler = MinMaxScaler()
    f1 = Scaler.fit_transform(f1.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f2 = Scaler.fit_transform(f2.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f3 = Scaler.fit_transform(f3.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f4 = Scaler.fit_transform(f4.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f5 = Scaler.fit_transform(f5.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f6 = Scaler.fit_transform(f6.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f7 = Scaler.fit_transform(f7.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f8 = Scaler.fit_transform(f8.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f9 = Scaler.fit_transform(f9.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f10 = Scaler.fit_transform(f10.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f11 = Scaler.fit_transform(f11.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f12 = Scaler.fit_transform(f12.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f13 = Scaler.fit_transform(f13.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f14 = Scaler.fit_transform(f14.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f15 = Scaler.fit_transform(f15.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f16 = Scaler.fit_transform(f16.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f17 = Scaler.fit_transform(f17.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f18 = Scaler.fit_transform(f18.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f19 = Scaler.fit_transform(f19.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f20 = Scaler.fit_transform(f20.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f21 = Scaler.fit_transform(f21.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f22 = Scaler.fit_transform(f22.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f23 = Scaler.fit_transform(f23.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f24 = Scaler.fit_transform(f24.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f25 = Scaler.fit_transform(f25.reshape(-1, 1))
    Scaler = MinMaxScaler()

    f26 = Scaler.fit_transform(f26.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f27 = Scaler.fit_transform(f27.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f28 = Scaler.fit_transform(f28.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f29 = Scaler.fit_transform(f29.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f30 = Scaler.fit_transform(f30.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f31 = Scaler.fit_transform(f31.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f32 = Scaler.fit_transform(f32.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f33 = Scaler.fit_transform(f33.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f34 = Scaler.fit_transform(f34.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f35 = Scaler.fit_transform(f35.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f36 = Scaler.fit_transform(f36.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f37 = Scaler.fit_transform(f37.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f38 = Scaler.fit_transform(f38.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f39 = Scaler.fit_transform(f39.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f40 = Scaler.fit_transform(f40.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f41 = Scaler.fit_transform(f41.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f42 = Scaler.fit_transform(f42.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f43 = Scaler.fit_transform(f43.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f44 = Scaler.fit_transform(f44.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f45 = Scaler.fit_transform(f45.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f46 = Scaler.fit_transform(f46.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f47 = Scaler.fit_transform(f47.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f48 = Scaler.fit_transform(f48.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f49 = Scaler.fit_transform(f49.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f50 = Scaler.fit_transform(f50.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f51 = Scaler.fit_transform(f51.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f52 = Scaler.fit_transform(f52.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f53 = Scaler.fit_transform(f53.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f54 = Scaler.fit_transform(f54.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f55 = Scaler.fit_transform(f55.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f56 = Scaler.fit_transform(f56.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f57 = Scaler.fit_transform(f57.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f58 = Scaler.fit_transform(f58.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f59 = Scaler.fit_transform(f59.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f60 = Scaler.fit_transform(f60.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f61 = Scaler.fit_transform(f61.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f62 = Scaler.fit_transform(f62.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f63 = Scaler.fit_transform(f63.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f64 = Scaler.fit_transform(f64.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f65 = Scaler.fit_transform(f65.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f66 = Scaler.fit_transform(f66.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f67 = Scaler.fit_transform(f67.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f68 = Scaler.fit_transform(f68.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f69 = Scaler.fit_transform(f69.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f70 = Scaler.fit_transform(f70.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f71 = Scaler.fit_transform(f71.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f72 = Scaler.fit_transform(f72.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f73 = Scaler.fit_transform(f73.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f74 = Scaler.fit_transform(f74.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f75 = Scaler.fit_transform(f75.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f76 = Scaler.fit_transform(f76.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f77 = Scaler.fit_transform(f77.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f78 = Scaler.fit_transform(f78.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f79 = Scaler.fit_transform(f79.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f80 = Scaler.fit_transform(f80.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f81 = Scaler.fit_transform(f81.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f82 = Scaler.fit_transform(f82.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f83 = Scaler.fit_transform(f83.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f84 = Scaler.fit_transform(f84.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f85 = Scaler.fit_transform(f85.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f86 = Scaler.fit_transform(f86.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f87 = Scaler.fit_transform(f87.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f88 = Scaler.fit_transform(f88.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f89 = Scaler.fit_transform(f89.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f90 = Scaler.fit_transform(f90.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f91 = Scaler.fit_transform(f91.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f92 = Scaler.fit_transform(f92.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f93 = Scaler.fit_transform(f93.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f94 = Scaler.fit_transform(f94.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f95 = Scaler.fit_transform(f95.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f96 = Scaler.fit_transform(f96.reshape(-1, 1))
    Scaler = MinMaxScaler()
    f99 = Scaler.fit_transform(f99.reshape(-1, 1))

    f1 = f1.reshape(2160, 4320)
    f2 = f2.reshape(2160, 4320)
    f3 = f3.reshape(2160, 4320)
    f4 = f4.reshape(2160, 4320)
    f5 = f5.reshape(2160, 4320)
    f6 = f6.reshape(2160, 4320)
    f7 = f7.reshape(2160, 4320)
    f8 = f8.reshape(2160, 4320)
    f9 = f9.reshape(2160, 4320)
    f10 = f10.reshape(2160, 4320)
    f11 = f11.reshape(2160, 4320)
    f12 = f12.reshape(2160, 4320)
    f13 = f13.reshape(2160, 4320)
    f14 = f14.reshape(2160, 4320)
    f15 = f15.reshape(2160, 4320)
    f16 = f16.reshape(2160, 4320)
    f17 = f17.reshape(2160, 4320)
    f18 = f18.reshape(2160, 4320)
    f19 = f19.reshape(2160, 4320)
    f20 = f20.reshape(2160, 4320)
    f21 = f21.reshape(2160, 4320)
    f22 = f22.reshape(2160, 4320)
    f23 = f23.reshape(2160, 4320)
    f24 = f24.reshape(2160, 4320)
    f25 = f25.reshape(2160, 4320)
    f26 = f26.reshape(2160, 4320)
    f27 = f27.reshape(2160, 4320)
    f28 = f28.reshape(2160, 4320)
    f29 = f29.reshape(2160, 4320)
    f30 = f30.reshape(2160, 4320)
    f31 = f31.reshape(2160, 4320)
    f32 = f32.reshape(2160, 4320)
    f33 = f33.reshape(2160, 4320)
    f34 = f34.reshape(2160, 4320)
    f35 = f35.reshape(2160, 4320)
    f36 = f36.reshape(2160, 4320)
    f37 = f37.reshape(2160, 4320)
    f38 = f38.reshape(2160, 4320)
    f39 = f39.reshape(2160, 4320)
    f40 = f40.reshape(2160, 4320)
    f41 = f41.reshape(2160, 4320)
    f42 = f42.reshape(2160, 4320)
    f43 = f43.reshape(2160, 4320)
    f44 = f44.reshape(2160, 4320)
    f45 = f45.reshape(2160, 4320)
    f46 = f46.reshape(2160, 4320)
    f47 = f47.reshape(2160, 4320)
    f48 = f48.reshape(2160, 4320)
    f49 = f49.reshape(2160, 4320)
    f50 = f50.reshape(2160, 4320)
    f51 = f51.reshape(2160, 4320)
    f52 = f52.reshape(2160, 4320)
    f53 = f53.reshape(2160, 4320)
    f54 = f54.reshape(2160, 4320)
    f55 = f55.reshape(2160, 4320)
    f56 = f56.reshape(2160, 4320)
    f57 = f57.reshape(2160, 4320)
    f58 = f58.reshape(2160, 4320)
    f59 = f59.reshape(2160, 4320)
    f60 = f60.reshape(2160, 4320)
    f61 = f61.reshape(2160, 4320)
    f62 = f62.reshape(2160, 4320)
    f63 = f63.reshape(2160, 4320)
    f64 = f64.reshape(2160, 4320)
    f65 = f65.reshape(2160, 4320)
    f66 = f66.reshape(2160, 4320)
    f67 = f67.reshape(2160, 4320)
    f68 = f68.reshape(2160, 4320)
    f69 = f69.reshape(2160, 4320)
    f70 = f70.reshape(2160, 4320)
    f71 = f71.reshape(2160, 4320)
    f72 = f72.reshape(2160, 4320)
    f73 = f73.reshape(2160, 4320)
    f74 = f74.reshape(2160, 4320)
    f75 = f75.reshape(2160, 4320)
    f76 = f76.reshape(2160, 4320)
    f77 = f77.reshape(2160, 4320)
    f78 = f78.reshape(2160, 4320)
    f79 = f79.reshape(2160, 4320)
    f80 = f80.reshape(2160, 4320)
    f81 = f81.reshape(2160, 4320)
    f82 = f82.reshape(2160, 4320)
    f83 = f83.reshape(2160, 4320)
    f84 = f84.reshape(2160, 4320)
    f85 = f85.reshape(2160, 4320)
    f86 = f86.reshape(2160, 4320)
    f87 = f87.reshape(2160, 4320)
    f88 = f88.reshape(2160, 4320)
    f89 = f89.reshape(2160, 4320)
    f90 = f90.reshape(2160, 4320)
    f91 = f91.reshape(2160, 4320)
    f92 = f92.reshape(2160, 4320)
    f93 = f93.reshape(2160, 4320)
    f94 = f94.reshape(2160, 4320)
    f95 = f95.reshape(2160, 4320)
    f96 = f96.reshape(2160, 4320)
    f99 = f99.reshape(2160, 4320)

    for i in range (len(LonRange)):
        if not i % 1== 0:
            LonRange[i]=10000
    for i in range (len(LatRange)):
        if not i % 1== 0:
            LatRange[i]=10000
    # Creating the training set (binning)
    VpBin = [[[] for _ in range(4320)] for _ in range(2160)]

    for i in range(len(Vp)):
        lon = Lons[i]
        lat = Lats[i]
        
        londiff = abs(LonRange - lon)
        lonidx = np.argmin(londiff)
        latdiff = abs(LatRange - lat)
        latidx = np.argmin(latdiff)
        
        VpBin[latidx][lonidx].append(Vp[i])


        
    BinnedVp = []
    f1_train = []
    f2_train = []
    f3_train = []
    f4_train = []
    f5_train = []
    f6_train = []
    f7_train = []
    f8_train = []
    f9_train = []
    f10_train = []
    f11_train = []
    f12_train = []
    f13_train = []
    f14_train = []
    f15_train = []
    f16_train = []
    f17_train = []
    f18_train = []
    f19_train = []
    f20_train = []
    f21_train = []
    f22_train = []
    f23_train = []
    f24_train = []
    f25_train = []
    f26_train = []
    f27_train = []
    f28_train = []
    f29_train = []
    f30_train = []
    f31_train = []
    f32_train = []
    f33_train = []
    f34_train = []
    f35_train = []
    f36_train = []
    f37_train = []
    f38_train = []
    f39_train = []
    f40_train = []
    f41_train = []
    f42_train = []
    f43_train = []
    f44_train = []
    f45_train = []
    f46_train = []
    f47_train = []
    f48_train = []
    f49_train = []
    f50_train = []
    f51_train = []
    f52_train = []
    f53_train = []
    f54_train = []
    f55_train = []
    f56_train = []
    f57_train = []
    f58_train = []
    f59_train = []
    f60_train = []
    f61_train = []
    f62_train = []
    f63_train = []
    f64_train = []
    f65_train = []
    f66_train = []
    f67_train = []
    f68_train = []
    f69_train = []
    f70_train = []
    f71_train = []
    f72_train = []
    f73_train = []
    f74_train = []
    f75_train = []
    f76_train = []
    f77_train = []
    f78_train = []
    f79_train = []
    f80_train = []
    f81_train = []
    f82_train = []
    f83_train = []
    f84_train = []
    f85_train = []
    f86_train = []
    f87_train = []
    f88_train = []
    f89_train = []
    f90_train = []
    f91_train = []
    f92_train = []
    f93_train = []
    f94_train = []
    f95_train = []
    f96_train = []
    f97_train = []
    f98_train = []
    f99_train = []
    f100_train = []
    
    BinnedLon = []
    BinnedLat = []

    #数据分箱处理，包括标签和特征
    for i in range(2160):
        for j in range(4320):
            VpBin[i][j] = np.mean(VpBin[i][j])
            if not np.isnan(VpBin[i][j]):
                BinnedLat.append(LatRange[i])
                BinnedLon.append(LonRange[j])
                BinnedVp.append(VpBin[i][j])
                f1_train.append(f1[i, j])
                f2_train.append(f2[i, j])
                f3_train.append(f3[i, j])
                f4_train.append(f4[i, j])
                f5_train.append(f5[i, j])
                f6_train.append(f6[i, j])
                f7_train.append(f7[i, j])
                f8_train.append(f8[i, j])
                f9_train.append(f9[i, j])
                f10_train.append(f10[i, j])
                f11_train.append(f11[i, j])
                f12_train.append(f12[i, j])
                f13_train.append(f13[i, j])
                f14_train.append(f14[i, j])
                f15_train.append(f15[i, j])
                f16_train.append(f16[i, j])
                f17_train.append(f17[i, j])
                f18_train.append(f18[i, j])
                f19_train.append(f19[i, j])
                f20_train.append(f20[i, j])
                f21_train.append(f21[i, j])
                f22_train.append(f22[i, j])
                f23_train.append(f23[i, j])
                f24_train.append(f24[i, j])
                f25_train.append(f25[i, j])
                f26_train.append(f26[i, j])
                f27_train.append(f27[i, j])
                f28_train.append(f28[i, j])
                f29_train.append(f29[i, j])
                f30_train.append(f30[i, j])
                f31_train.append(f31[i, j])
                f32_train.append(f32[i, j])
                f33_train.append(f33[i, j])
                f34_train.append(f34[i, j])
                f35_train.append(f35[i, j])
                f36_train.append(f36[i, j])
                f37_train.append(f37[i, j])
                f38_train.append(f38[i, j])
                f39_train.append(f39[i, j])
                f40_train.append(f40[i, j])
                f41_train.append(f41[i, j])
                f42_train.append(f42[i, j])
                f43_train.append(f43[i, j])
                f44_train.append(f44[i, j])
                f45_train.append(f45[i, j])
                f46_train.append(f46[i, j])
                f47_train.append(f47[i, j])
                f48_train.append(f48[i, j])
                f49_train.append(f49[i, j])
                f50_train.append(f50[i, j])
                f51_train.append(f51[i, j])
                f52_train.append(f52[i, j])
                f53_train.append(f53[i, j])
                f54_train.append(f54[i, j])
                f55_train.append(f55[i, j])
                f56_train.append(f56[i, j])
                f57_train.append(f57[i, j])
                f58_train.append(f58[i, j])
                f59_train.append(f59[i, j])
                f60_train.append(f60[i, j])
                f61_train.append(f61[i, j])
                f62_train.append(f62[i, j])
                f63_train.append(f63[i, j])
                f64_train.append(f64[i, j])
                f65_train.append(f65[i, j])
                f66_train.append(f66[i, j])
                f67_train.append(f67[i, j])
                f68_train.append(f68[i, j])
                f69_train.append(f69[i, j])
                f70_train.append(f70[i, j])
                f71_train.append(f71[i, j])
                f72_train.append(f72[i, j])
                f73_train.append(f73[i, j])
                f74_train.append(f74[i, j])
                f75_train.append(f75[i, j])
                f76_train.append(f76[i, j])
                f77_train.append(f77[i, j])
                f78_train.append(f78[i, j])
                f79_train.append(f79[i, j])
                f80_train.append(f80[i, j])
                f81_train.append(f81[i, j])
                f82_train.append(f82[i, j])
                f83_train.append(f83[i, j])
                f84_train.append(f84[i, j])
                f85_train.append(f85[i, j])
                f86_train.append(f86[i, j])
                f87_train.append(f87[i, j])
                f88_train.append(f88[i, j])
                f89_train.append(f89[i, j])
                f90_train.append(f90[i, j])
                f91_train.append(f91[i, j])
                f92_train.append(f92[i, j])
                f93_train.append(f93[i, j])
                f94_train.append(f94[i, j])
                f95_train.append(f95[i, j])
                f96_train.append(f96[i, j])
                f99_train.append(f99[i, j])

    
    BinnedLon = np.array(BinnedLon)
    BinnedLat = np.array(BinnedLat)
    BinnedVp = np.array(BinnedVp)
    f1_train = np.array(f1_train)
    f2_train = np.array(f2_train)
    f3_train = np.array(f3_train)
    f4_train = np.array(f4_train)
    f5_train = np.array(f5_train)
    f6_train = np.array(f6_train)
    f7_train = np.array(f7_train)
    f8_train = np.array(f8_train)
    f9_train = np.array(f9_train)
    f10_train = np.array(f10_train)
    f11_train = np.array(f11_train)
    f12_train = np.array(f12_train)
    f13_train = np.array(f13_train)
    f14_train = np.array(f14_train)
    f15_train = np.array(f15_train)
    f16_train = np.array(f16_train)
    f17_train = np.array(f17_train)
    f18_train = np.array(f18_train)
    f19_train = np.array(f19_train)
    f20_train = np.array(f20_train)
    f21_train = np.array(f21_train)
    f22_train = np.array(f22_train)
    f23_train = np.array(f23_train)
    f24_train = np.array(f24_train)
    f25_train = np.array(f25_train)
    f26_train = np.array(f26_train)
    f27_train = np.array(f27_train)
    f28_train = np.array(f28_train)
    f29_train = np.array(f29_train)
    f30_train = np.array(f30_train)
    f31_train = np.array(f31_train)
    f32_train = np.array(f32_train)
    f33_train = np.array(f33_train)
    f34_train = np.array(f34_train)
    f35_train = np.array(f35_train)
    f36_train = np.array(f36_train)
    f37_train = np.array(f37_train)
    f38_train = np.array(f38_train)
    f39_train = np.array(f39_train)
    f40_train = np.array(f40_train)
    f41_train = np.array(f41_train)
    f42_train = np.array(f42_train)
    f43_train = np.array(f43_train)
    f44_train = np.array(f44_train)
    f45_train = np.array(f45_train)
    f46_train = np.array(f46_train)
    f47_train = np.array(f47_train)
    f48_train = np.array(f48_train)
    f49_train = np.array(f49_train)
    f50_train = np.array(f50_train)
    f51_train = np.array(f51_train)
    f52_train = np.array(f52_train)
    f53_train = np.array(f53_train)
    f54_train = np.array(f54_train)
    f55_train = np.array(f55_train)
    f56_train = np.array(f56_train)
    f57_train = np.array(f57_train)
    f58_train = np.array(f58_train)
    f59_train = np.array(f59_train)
    f60_train = np.array(f60_train)
    f61_train = np.array(f61_train)
    f62_train = np.array(f62_train)
    f63_train = np.array(f63_train)
    f64_train = np.array(f64_train)
    f65_train = np.array(f65_train)
    f66_train = np.array(f66_train)
    f67_train = np.array(f67_train)
    f68_train = np.array(f68_train)
    f69_train = np.array(f69_train)
    f70_train = np.array(f70_train)
    f71_train = np.array(f71_train)
    f72_train = np.array(f72_train)
    f73_train = np.array(f73_train)
    f74_train = np.array(f74_train)
    f75_train = np.array(f75_train)
    f76_train = np.array(f76_train)
    f77_train = np.array(f77_train)
    f78_train = np.array(f78_train)
    f79_train = np.array(f79_train)
    f80_train = np.array(f80_train)
    f81_train = np.array(f81_train)
    f82_train = np.array(f82_train)
    f83_train = np.array(f83_train)
    f84_train = np.array(f84_train)
    f85_train = np.array(f85_train)
    f86_train = np.array(f86_train)
    f87_train = np.array(f87_train)
    f88_train = np.array(f88_train)
    f89_train = np.array(f89_train)
    f90_train = np.array(f90_train)
    f91_train = np.array(f91_train)
    f92_train = np.array(f92_train)
    f93_train = np.array(f93_train)
    f94_train = np.array(f94_train)
    f95_train = np.array(f95_train)
    f96_train = np.array(f96_train)
    f97_train = np.array(f97_train)
    f98_train = np.array(f98_train)
    f99_train = np.array(f99_train)
    f100_train = np.array(f100_train)
    '''
    ______________________________
    The feature selection results of the Spatial Constrained Deep Forest algorithm 
        show that 20 of them achieved the best performance.
    ______________________________
    '''
    X_all = np.hstack((f74_train.reshape(-1, 1), f4_train.reshape(-1, 1), f67_train.reshape(-1, 1),
                       f69_train.reshape(-1, 1), f39_train.reshape(-1, 1), f70_train.reshape(-1, 1),
                       f57_train.reshape(-1, 1), f18_train.reshape(-1, 1), f80_train.reshape(-1, 1),
                       f9_train.reshape(-1, 1), f71_train.reshape(-1, 1), f77_train.reshape(-1, 1),
                       f37_train.reshape(-1, 1), f27_train.reshape(-1, 1), f85_train.reshape(-1, 1),
                       f14_train.reshape(-1, 1), f63_train.reshape(-1, 1), f78_train.reshape(-1, 1),
                       f54_train.reshape(-1, 1), f46_train.reshape(-1, 1), f64_train.reshape(-1, 1),
                       f60_train.reshape(-1, 1), f29_train.reshape(-1, 1), f5_train.reshape(-1, 1),
                       f13_train.reshape(-1, 1), f72_train.reshape(-1, 1), f66_train.reshape(-1, 1),
                       f45_train.reshape(-1, 1), f59_train.reshape(-1, 1), f47_train.reshape(-1, 1),
                       f19_train.reshape(-1, 1), f73_train.reshape(-1, 1), f87_train.reshape(-1, 1),
                       f20_train.reshape(-1, 1), f68_train.reshape(-1, 1), f11_train.reshape(-1, 1),
                       f35_train.reshape(-1, 1), f8_train.reshape(-1, 1), f41_train.reshape(-1, 1),
                       f38_train.reshape(-1, 1), f81_train.reshape(-1, 1), f82_train.reshape(-1, 1),
                       f51_train.reshape(-1, 1), f44_train.reshape(-1, 1), f55_train.reshape(-1, 1),
                       f65_train.reshape(-1, 1), f92_train.reshape(-1, 1), f26_train.reshape(-1, 1),
                       f83_train.reshape(-1, 1), f16_train.reshape(-1, 1), f79_train.reshape(-1, 1),
                       f89_train.reshape(-1, 1), f23_train.reshape(-1, 1), f34_train.reshape(-1, 1),
                       f36_train.reshape(-1, 1), f10_train.reshape(-1, 1), f30_train.reshape(-1, 1),
                       f6_train.reshape(-1, 1), f43_train.reshape(-1, 1), f33_train.reshape(-1, 1),
                       f25_train.reshape(-1, 1), f32_train.reshape(-1, 1), f88_train.reshape(-1, 1),
                       f94_train.reshape(-1, 1), f40_train.reshape(-1, 1), f3_train.reshape(-1, 1),
                       f62_train.reshape(-1, 1), f21_train.reshape(-1, 1), f42_train.reshape(-1, 1),
                       f90_train.reshape(-1, 1), f12_train.reshape(-1, 1), f15_train.reshape(-1, 1),
                       f52_train.reshape(-1, 1), f22_train.reshape(-1, 1), f50_train.reshape(-1, 1),
                       f48_train.reshape(-1, 1), f84_train.reshape(-1, 1), f7_train.reshape(-1, 1),
                       f53_train.reshape(-1, 1), f17_train.reshape(-1, 1), f58_train.reshape(-1, 1),
                       f61_train.reshape(-1, 1), f31_train.reshape(-1, 1), f93_train.reshape(-1, 1),
                       f56_train.reshape(-1, 1), f24_train.reshape(-1, 1), f28_train.reshape(-1, 1),
                       f49_train.reshape(-1, 1), f76_train.reshape(-1, 1), f86_train.reshape(-1, 1),
                       f75_train.reshape(-1, 1), f91_train.reshape(-1, 1),))

    sio.savemat('BinnedData/LabeledData.mat', {'X_all': X_all,
                                                                      'BinnedVp': BinnedVp,
                                                                      'BinnedLon': BinnedLon,
                                                                          'BinnedLat': BinnedLat})