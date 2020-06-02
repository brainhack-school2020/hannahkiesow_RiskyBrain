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


# split the data into training and test set 


# CCA for feature extraction 
from sklearn.cross_decomposition import CCA

X = DECONF_DMN_vols
Y = DECONF_FSL_vols

n_keep = 10
model_cca = CCA(n_components=n_keep, scale=False)
model_cca.fit(X, Y)
X_c, Y_c = model_cca.transform(X, Y)

# get correlations of modes 
from scipy.stats import pearsonr
correlations = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
    zip(model_cca.x_scores_.T, model_cca.y_scores_.T)])




# visualize the components

# Variate X (Social Brain Regions)
import matplotlib.pyplot as plt
import seaborn as sns

# function that gives you plots for X and Y variates 
def plot_CCA(n_CCA_to_plot, grid_n, variate_weights, labels):
    n_keep = n_CCA_to_plot
    for n in range(n_keep):
    plot = plt.figure(figsize=(10, 7))

    grid = np.zeros(grid_n , grid_n)
    
    triu_mask = np.triu(np.ones_like(grid, dtype=np.bool))
    
    weights = np.tril(variate_weights)
    
    TH = 0.00
    variate_weights[(variate_weights < TH) & (variate_weights > -TH)] = 0

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    ax = sns.heatmap(data=variate_weights, mask=triu_mask, cbar=True, linewidths=.5,
                     vmin=-0.5, vmax=0.5, center=0,
                     cmap=cmap, square=True, 
                     cbar_kws={"shrink": .5})
    ax.set_yticks(np.arange(len(rois)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)
    plt.title('Canonical component %i in Social Brain subnodes' % (n + 1))
    plt.tight_layout()
    return plot # TO DO: debug function!




n_keep = 3
for n in range(n_keep):
    plt.figure(figsize=(10, 7))

    grid = np.zeros((len(rois), len(rois)))
    
    triu_mask = np.triu(np.ones_like(grid, dtype=np.bool))
    
    X_weights = np.tril(model_cca.x_loadings_[:, n])
    
    TH = 0.00
    X_weights[(X_weights < TH) & (X_weights > -TH)] = 0

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    ax = sns.heatmap(data=X_weights, mask=triu_mask, cbar=True, linewidths=.5,
                     vmin=-0.5, vmax=0.5, center=0,
                     cmap=cmap, square=True, 
                     cbar_kws={"shrink": .5})
    ax.set_yticks(np.arange(len(rois)))
    ax.set_xticklabels(roi_names, rotation=90)
    ax.set_yticklabels(roi_names, rotation=0)
    plt.title('Canonical component %i in Social Brain subnodes' % (n + 1))
    plt.tight_layout()


# Variate Y (FSL Atlas regions)




# permutation tests 
n_permutations = 1000
perm_rs = np.random.RandomState(42)
perm_Rs = []
n_except = 0
for i_iter in range(n_permutations):
    print(i_iter + 1)

    Y_netnet_perm = np.array([perm_rs.permutation(sub_row) for sub_row in Y])

    # Y_netnet_perm = np.array([perm_rs.permutation(sub_row) for sub_row in cur_Y.T])

    # same procedure, only with permuted subjects on the right side
    try:
        perm_cca = CCA(n_components=n_keep, scale=False)

        # perm_inds = np.arange(len(Y_netmet))
        # perm_rs.shuffle(perm_inds)
        # perm_cca.fit(X_nodenode, Y_netnet[perm_inds, :])
        perm_cca.fit(X, Y)

        perm_R = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
            zip(perm_cca.x_scores_.T, perm_cca.y_scores_.T)])
        perm_Rs.append(perm_R)
    except:
        n_except += 1
        perm_Rs.append(np.zeros(n_keep))
perm_Rs = np.array(perm_Rs)

pvals = []
for i_coef in range(n_keep):
    cur_pval = (1. + np.sum(perm_Rs[1:, 0] > correlations[i_coef])) / n_permutations
    pvals.append(cur_pval)
    print (cur_pval)
pvals = np.array(pvals)
print('%i CCs are significant at p<0.05' % np.sum(pvals < 0.05))
print('%i CCs are significant at p<0.01' % np.sum(pvals < 0.01))
print('%i CCs are significant at p<0.001' % np.sum(pvals < 0.001))

final_n = 1




