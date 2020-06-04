"""
This script produces the social brain atlas renderings in grayscale (produces a simple atlas) 

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


seed_dict = collections.OrderedDict([
('AI_L', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_lAI_vox200.nii.gz'),
('AI_R', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_rAI_vox200.nii.gz'),
('AM_L', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_lAM_vox200.nii.gz'),
('AM_R', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_rAM_vox200.nii.gz'),
('FG_L', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_lFFA_vox200.nii.gz'),
('FG_R', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_rFFA_vox200.nii.gz'),
('FP', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_FP_vox200.nii.gz'),
('HC_L', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_lHC_vox200.nii.gz'),
('HC_R', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_rHC_vox200.nii.gz'),
('IFG_L', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_lIFG_vox200.nii.gz'),
('IFG_R', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_rIFG_vox200.nii.gz'),
('MTG_L', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_lMTG_vox200.nii.gz'),
('MTG_R', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_rMTG_vox200.nii.gz'),
('MTV5_L', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_lMTV5_vox200.nii.gz'),
('MTV5_R', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_rMTV5_vox200.nii.gz'),
('NAC_L', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_lNAcc_vox200.nii.gz'),
('NAC_R', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_rNAcc_vox200.nii.gz'),
('PCC', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_PCC_vox200.nii.gz'),
('Prec', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_PCu_vox200.nii.gz'),
('SMA_L', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_lSMA_vox200.nii.gz'),
('SMA_R', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_rSMA_vox200.nii.gz'),
('SMG_L', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_lIPL_vox200.nii.gz'),
('SMG_R', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_rIPL_vox200.nii.gz'),
('TPJ_L', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_lTPJ_vox200.nii.gz'),
('TPJ_R', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_rTPJ_vox200.nii.gz'),
('TP_L', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_lTP_vox200.nii.gz'),
('TP_R', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_rTP_vox200.nii.gz'),
('aMCC', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_aMCC_vox200.nii.gz'),
('dmPFC', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_dmPFC_vox200.nii.gz'),
('pMCC', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_pMCC_vox200.nii.gz'),
('pSTS_L', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_lpSTS_vox200.nii.gz'),
('pSTS_R', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_rpSTS_vox200.nii.gz'),
('rACC', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_rACC_vox200.nii.gz'),
('vmPFC', r'SB/socialbrain_seeds_sym/socialbrain_seeds_sym/seed_vmPFC_vox200.nii.gz')])

from nilearn.input_data import NiftiSpheresMasker
from nilearn.plotting import find_xyz_cut_coords

tmp_nii = nib.load('/Users/hannah/SB/rcolin_bin.nii.gz')
target_aff, target_shape = tmp_nii.affine, tmp_nii.shape

colin_nii = resample_img('/Users/hannah/SB/colin.nii', target_affine=target_aff,  # map ROI to data space
        target_shape=target_shape, interpolation='nearest')
from nilearn.image import math_img
colin_nii = math_img('img > 0', img=colin_nii)


# get NiftiLabels-compatible summary seed nifti
social_atlas = None
roi_names = []
seed_coords = []
for i_seed, (key, value) in enumerate(seed_dict.items()):
    cur_roi_path = os.path.join(os.getcwd(), value)
    seed_nii = nib.load(cur_roi_path)
    seed_nii = resample_img(seed_nii, target_affine=target_aff,  # map ROI to data space
            target_shape=target_shape, interpolation='nearest')
    seed_coord = find_xyz_cut_coords(seed_nii)
    if social_atlas is None:
        social_atlas = np.zeros(seed_nii.shape)
    print(key)
    print(seed_coord)
    inds = seed_nii.get_data() > 0.1
    social_atlas[inds] = i_seed + 1  # current cluster label
    roi_names.append(key)
    seed_coords.append(seed_coord)
social_nii = nib.Nifti1Image(social_atlas, seed_nii.get_affine())

list(zip(roi_names, seed_coords))  # raw output as sanity check





net1_L = ['FG_L', 'pSTS_L', 'MTV5_L']
net1_R = ['FG_R', 'pSTS_R', 'MTV5_R']
net2_L = ['AM_L', 'HC_L', 'vmPFC', 'NAC_L', 'rACC']
net2_R = ['AM_R', 'HC_R', 'vmPFC', 'NAC_R', 'rACC']
net3_L = ['aMCC',  'AI_L', 'SMG_L', 'SMA_L',  'IFG_L',  'Cereb_L']
net3_R = ['aMCC',  'AI_R', 'SMG_R', 'SMA_R',  'IFG_R',  'Cereb_R']
net4_L = ['FP', 'dmPFC',  'PCC',  'TPJ_L',  'Prec',  'MTG_L',  'TP_L', 'pMCC']
net4_R = ['FP', 'dmPFC',  'PCC',  'TPJ_R',  'Prec',  'MTG_R',  'TP_R', 'pMCC']

roi_groups = [net1_L, net1_R, net2_L, net2_R, net3_L, net3_R, net4_L, net4_R]


# super old code - but it works! need to redo

network1_L, network1_R, network2_L, network2_R, network3_L, network3_R, network4_L, network4_R = [], [], [], [], [], [], [], []
for roi, coord in zip(roi_names, seed_coords): 
    while roi in net1_L: 
        print('network1 LEFT')
        network1_L.append(coord)
        break
    while roi in net1_R:
        print('network1 RIGHT')
        network1_R.append(coord)
        break
    while roi in net2_L:
        network2_L.append(coord)
        break
    while roi in net2_R:
        print('network2 RIGHT')
        network2_R.append(coord)
        break
    while roi in net3_L: 
        print('network3 LEFT')
        network3_L.append(coord)
        break
    while roi in net3_R:
        print('network3 RIGHT')
        network3_R.append(coord)
        break
    while roi in net4_L: 
        print('network4 LEFT')
        network4_L.append(coord)
        break
    while roi in net4_R:
        print('network4 RIGHT')
        network4_R.append(coord)
        break

zip(network1_L, network1_R, network2_L, network2_R, network3_L, network3_R, network4_L, network4_R)  # raw output as sanity check






color = '#4D4D4D' #gray
subject_id = "fsaverage"
hemi = 'lh'
surf = 'inflated'
color = '#FF007F'
alpha=0.75

# whole brain 
brain = Brain(subject_id, "both", 'inflated', cortex=("gray", -2, 7, True), background='black')
for roi_name, seed_coord in zip(roi_names, seed_coords):
    print(roi_name)
    print(seed_coord)
    x = seed_coord[0]
    if x > 0 or np.abs(x) < 6:
        #brain.add_foci(seed_coord, map_surface="white", color=color,
        #    scale_factor=1.6, hemi='rh', alpha=alpha)
        brain.add_foci(network1_L, map_surface="white", hemi='lh', color="#F5D300", scale_factor=1.6) # (Visual/Sensory)
        brain.add_foci(network2_L, map_surface="white", hemi='lh', color="#09FBD3", scale_factor=1.6) # (Limbic)
        brain.add_foci(network3_L, map_surface="white", hemi='lh', color="#FE53BB", scale_factor=1.6) # (Intermediate)
        brain.add_foci(network4_L, map_surface="white", hemi='lh', color="#08F7FE", scale_factor=1.6) #
    if x < 0 or np.abs(x) < 6:
        brain.add_foci(seed_coord, map_surface="white", color=color,
            scale_factor=1.6, hemi='lh', alpha=alpha)
        brain.add_foci(network1_R, map_surface="white", hemi='rh', color="#F5D300", scale_factor=1.6) # (Visual/Sensory)
        brain.add_foci(network2_R, map_surface="white", hemi='rh', color="#09FBD3", scale_factor=1.6) # (Limbic)
        brain.add_foci(network3_R, map_surface="white", hemi='rh', color="#FE53BB", scale_factor=1.6) # (Intermediate)
        brain.add_foci(network4_R, map_surface="white", hemi='rh', color="#08F7FE", scale_factor=1.6) # (Higher Associative)

# the actual animation:
brain.animate(["m"] * 3, n_steps=100) 
