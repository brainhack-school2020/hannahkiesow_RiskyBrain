import joblib
import numpy as np
import pandas as pd 
DECONF = True

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


ukbb_target = ukbb.copy()[b_inds_ukbb]


# extract sMRI data
ukbb_sMRI = ukbb_target.loc[:, '25782-2.0':'25892-2.0']  # FSL atlas without Diederichsen cerebellar atlas

# check for missing values 
np.isnan(DMN_vols).sum() # none here

ukbb_sMRI[ukbb_sMRI.isnull().any(axis=1)] # 2 rows with complete missing values


# remove missing values 
ukbb_sMRI = ukbb_sMRI.copy().dropna()

# drop participants missing from ukbb_sMRI also from DMN_vols
DMN_vols = np.delete(DMN_vols, [4411, 9834], 0)

# sanity checks to make sure missing values are gone
ukbb_sMRI[ukbb_sMRI.isnull().any(axis=1)]
np.isnan(DMN_vols).sum() 

assert DMN_vols.shape[0] == ukbb_sMRI.shape[0]


# standardize volumes
from sklearn.preprocessing import StandardScaler
DMN_vols_standardized = StandardScaler().fit_transform(DMN_vols)
FSL_vols_standardized = StandardScaler().fit_transform(np.array(ukbb_sMRI))


# deconfound for head size and BMI 
ukbb_target_no_nans = ukbb_target.drop([4411, 9834])

head_size = StandardScaler().fit_transform(np.nan_to_num(ukbb_target_no_nans['25006-2.0'].values[:, None]))  # Volume of grey matter
body_mass = StandardScaler().fit_transform(np.nan_to_num(ukbb_target_no_nans['21001-0.0'].values[:, None]))  # BMI
conf_mat = np.hstack([
    np.atleast_2d(head_size), np.atleast_2d(body_mass)])

if DECONF == True:
    from nilearn.signal import clean

    print('Deconfounding BMI & grey-matter space!')
    DECONF_DMN_vols = clean(DMN_vols_standardized, confounds=conf_mat, detrend=False, standardize=False)
    DECONF_FSL_vols = clean(FSL_vols_standardized, confounds=conf_mat, detrend=False, standardize=False)


# get atlases
from nilearn import datasets as ds
HO_atlas_cort = ds.fetch_atlas_harvard_oxford('cort-maxprob-thr50-1mm', symmetric_split=True)
HO_atlas_sub = ds.fetch_atlas_harvard_oxford('sub-maxprob-thr50-1mm', symmetric_split=True)


# CCA for feature extraction 
from sklearn.cross_decomposition import CCA

X = DECONF_DMN_vols
Y = DECONF_FSL_vols

n_keep = 36
n_permutations = 1000

model_cca = CCA(n_components=n_keep, scale=False)
model_cca.fit(X, Y)
X_c, Y_c = model_cca.transform(X, Y)








