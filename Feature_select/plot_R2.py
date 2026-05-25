import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd

matcontent = sio.loadmat('RMSE_SP_DF.mat')

R2_change=[]
R2_choose=matcontent['RMSE']


df = pd.read_excel('predictors_for-rmse_SPDF.xlsx')
r2 = df['R2'].values
r2_random = df['R2_random'][0]
r2 = sorted(r2,reverse = False)
R2_single=r2






r2_index = np.array(np.where(R2_single)).reshape(-1,1)
font ={'size': 14, 'family' : 'Arial','weight' : 'normal'}
fig = plt.figure(figsize=(9,6))
plt.scatter(range(1,93),R2_single,label='Individual Grid',c='b',s=20)
#plt.plot(range(1,93),R2_noise*np.ones(92).reshape(-1,1),'-',c='blue',label='Noise Grid')
plt.xlabel('Index',font)
plt.ylabel('RMSE',font)
R2_choose=R2_choose.reshape(-1,1)
plt.scatter(range(2,92),R2_choose[1:91],label='Feature',s=20)
plt.scatter(range(1,93),R2_choose,label='Ensemble Grids',s=20)
plt.legend(fontsize=10)
plt.title('Feature Selection', fontsize=12)
plt.grid(True)
plt.xscale('log')
#plt.savefig(r'figure\Index_R2.png',dpi=1200)
#sio.savemat('Feature_Selection', {'R2_Ensemble_encoder':R2_change })
plt.show()