import joblib
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler


OUT_DIR = ('/Users/hannah/hannahkiesow_RiskyBrain')


# load social brain volumes
dump_path = '/Users/hannah/dump_sMRI_socialbrain_sym_r2.5_s5'
T1_subnames, DMN_vols, rois = joblib.load(dump_path)
rois = np.array(rois)
T1_subnames_int = np.array([np.int(nr) for nr in T1_subnames], dtype=np.int64)
roi_names = np.array(rois)


# load UKBB
ukbb_path = '/Users/hannah/SB/ukb_add1_holmes_merge_brain.csv'
if 'ukbb' not in locals():
    ukbb = pd.read_csv(ukbb_path)
else:
    print('Database is already in memory!')



# match ukbb subjects to social brain volumes 
inds_mri = []
source_array = T1_subnames_int
for _, subject in enumerate(ukbb.eid):
    i_found = np.where(subject == source_array)[0]
    if len(i_found) == 0:
        continue
    inds_mri.append(i_found[0])  # take first found subject
b_inds_ukbb = np.in1d(ukbb.eid, source_array[inds_mri])

print('%i matched matrices between grey matter data and UKBB found!' % np.sum(
        source_array[inds_mri] == ukbb.eid[b_inds_ukbb]))


# keep only the matched subjects
T1_subnames = T1_subnames[inds_mri]
T1_subnames_int = T1_subnames_int[inds_mri]
DMN_vols = DMN_vols[inds_mri]
assert np.sum(T1_subnames_int == ukbb.eid[b_inds_ukbb].values) == len(inds_mri)


ukbb_target = ukbb[b_inds_ukbb]


# extract sMRI data
ukbb_sMRI = ukbb_target.loc[:, '25782-2.0':'25892-2.0']  # FSL atlas without Diederichsen cerebellar atlas


# standardize volumes
DMN_vols_standardized = StandardScaler().fit_transform(DMN_vols)
whole_brain_standardized = StandardScaler().fit_transform(np.array(ukbb_sMRI))


