import os
import numpy as np
import pandas as pd
import joblib
import glob
import matplotlib.pylab as plt
import seaborn as sns
import pymc3 as pm
import ptitprince as pt


%matplotlib
sns.set_style("white")    

OUT_DIR = ('/Users/hannah/hannahkiesow_RiskyBrain')


# load UKBB
ukbb_path = '/Users/hannah/SB/ukb_add1_holmes_merge_brain.csv'
if 'ukbb' not in locals():
    ukbb = pd.read_csv(ukbb_path)
else:
    print('Database is already in memory!')


keep = ['31-0.0', '21022-0.0', '2040-0.0']
ukbb_dem = ukbb.copy()[keep].dropna()
ukbb_dem = ukbb_dem[ukbb_dem['2040-0.0'] >= 0]     




sex='31-0.0'; age='21022-0.0'; risk='2040-0.0'
sex_colors = sns.set_palette(['#ff019c', '#F5D300'])

f, ax = plt.subplots(figsize=(8, 8))
pt.RainCloud(x = risk, y = age, data = ukbb_dem,
                ax = ax, hue=sex, palette=sex_colors, orient='h',
                dodge=True, point_size=1.0, linewidth=0.2,
                width_viol=1.3, width_box=0.25,
                alpha=0.60)
plt.tight_layout()    

plt.savefig('%s/raincloud.png' % (OUT_DIR), dpi=600)



