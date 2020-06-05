"""
This script produces the social brain atlas renderings 

paste the next two lines into your terminal before you run ipython:
export FREESURFER_HOME=/Applications/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
"""



from surfer import Brain
%matplotlib
import os
import numpy as np
import pandas as pd 
from collections import OrderedDict
import collections
from nilearn.image import resample_img
import nibabel as nib

OUT_DIR = ('/Users/hannah/hannahkiesow_RiskyBrain')



color = '#4D4D4D' #gray
subject_id = "fsaverage"
hemi = 'lh'
surf = 'inflated'
color = '#FF007F'
alpha=0.75

# whole brain 
brain = Brain(subject_id, "both", 'inflated', cortex=("gray", -2, 7, True), background='black', alpha=0.6)
for roi_name, seed_coord in zip(roi_names, seed_coords):
    print(roi_name)
    print(seed_coord)
    x = seed_coord[0]
    if x > 0 or np.abs(x) < 6:
        #brain.add_foci(seed_coord, map_surface="white", color=color,
        #    scale_factor=1.6, hemi='rh', alpha=alpha)
        brain.add_foci(network1_L, map_surface="white", hemi='lh', color="#F5D300", scale_factor=1.6) # Visual/Sensory
        brain.add_foci(network2_L, map_surface="white", hemi='lh', color="#09FBD3", scale_factor=1.6) # Limbic
        brain.add_foci(network3_L, map_surface="white", hemi='lh', color="#FE53BB", scale_factor=1.6) # Intermediate
        brain.add_foci(network4_L, map_surface="white", hemi='lh', color="#A117F2", scale_factor=1.6) # Higher Associative
    if x < 0 or np.abs(x) < 6:
        #brain.add_foci(seed_coord, map_surface="white", color=color,
        #    scale_factor=1.6, hemi='lh', alpha=alpha)
        brain.add_foci(network1_R, map_surface="white", hemi='rh', color="#F5D300", scale_factor=1.6) # Visual/Sensory
        brain.add_foci(network2_R, map_surface="white", hemi='rh', color="#09FBD3", scale_factor=1.6) # Limbic
        brain.add_foci(network3_R, map_surface="white", hemi='rh', color="#FE53BB", scale_factor=1.6) # Intermediate
        brain.add_foci(network4_R, map_surface="white", hemi='rh', color="#A117F2", scale_factor=1.6) # Higher Associative

# the actual animation:
brain.animate(["m"] * 3, n_steps=100) 
