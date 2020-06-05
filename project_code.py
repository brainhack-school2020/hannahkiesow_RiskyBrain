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


DMN_vols = pd.DataFrame(DMN_vols, columns=rois)


# extract sMRI data + risk-taking variable (what we will predict)
# also deconfounding variables
ukbb_sMRI = ukbb_target.copy().loc[:, '25782-2.0':'25892-2.0']  # FSL atlas without Diederichsen cerebellar atlas


 
ukbb_risk =  ukbb_target.copy()['2040-0.0']

confounds = ['25006-2.0', '21001-0.0']
ukbb_confounds = ukbb_target.copy()[confounds]

sMRI = pd.concat([DMN_vols, ukbb_sMRI], axis=1)     


# check for missing values 

sMRI[sMRI.isnull().any(axis=1)] # 2 rows with complete missing values

ukbb_risk[ukbb_risk.isnull()] # 3 missing values 


# impute NaN, -1.0 (does not know) and -3.0 (prefer not to answer)
# do this only for the risk-taking target

np.random.seed(0)
def my_impute(arr):
    print('Replacing %i NaN values!' % np.sum(np.isnan(arr)))
    arr = np.array(arr)
    b_nan = np.isnan(arr)
    b_negative = arr < 0
    b_bad = b_nan | b_negative

    arr[b_bad] = np.random.choice(arr[~b_bad], np.sum(b_bad))
    arr = pd.Series(arr)
    return arr

ukbb_risk = my_impute(ukbb_risk)


# remove missing values 
sMRI = sMRI.dropna()

# drop participants missing from ukbb_sMRI also from risk and confound dataframes
ukbb_risk = ukbb_risk.drop(labels=[4411, 9834])   
ukbb_confounds = ukbb_confounds.drop(labels=[4411, 9834])   

# sanity checks to make sure missing values are gone
sMRI[sMRI.isnull().any(axis=1)]
ukbb_risk[ukbb_risk.isnull()]

assert sMRI.shape[0] == ukbb_risk.shape[0] == ukbb_confounds.shape[0]

# put confounds together with vols 
sMRI_conf = pd.concat([sMRI, ukbb_confounds], axis=1)  

# split the data into training and test set 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    sMRI_conf, ukbb_risk, test_size=0.25, random_state=42)

X_train_sMRI = X_train.iloc[:, :-2]
X_test_sMRI = X_test.iloc[:, :-2]

X_train_conf = X_train.iloc[:, -2:]
X_test_conf = X_test.iloc[:, -2:]

# standardize volumes
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_SS = sc.fit_transform(X_train_sMRI)
X_test_SS = sc.transform(X_test_sMRI)


# deconfound for head size and BMI 
head_size = sc.fit_transform(np.nan_to_num(X_train_conf['25006-2.0'].values[:, None]))  # Volume of grey matter
body_mass = sc.fit_transform(np.nan_to_num(X_train_conf['21001-0.0'].values[:, None]))  # BMI
conf_mat = np.hstack([
    np.atleast_2d(head_size), np.atleast_2d(body_mass)])

if DECONF == True:
    from nilearn.signal import clean

    print('Deconfounding BMI & grey-matter space!')
    X_train_DECONF = clean(X_train_SS, confounds=conf_mat, detrend=False, standardize=False)
    


# get atlases
from nilearn import datasets as ds
HO_atlas_cort = ds.fetch_atlas_harvard_oxford('cort-maxprob-thr50-1mm', symmetric_split=True)
HO_atlas_sub = ds.fetch_atlas_harvard_oxford('sub-maxprob-thr50-1mm', symmetric_split=True)



# CCA for feature extraction 
from sklearn.cross_decomposition import CCA

X = X_train_DECONF[:, :36]
Y = X_train_DECONF[:, 36:]

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
%matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# function that gives you plots for X and Y variates 

# loadings = a variable's relationship to its own set
#   weights = variable's relationship to the other set, collinearity not considered
#   scores = variable's relationship to the other set, collinearity considered (I am guessing)
X_loadings = model_cca.x_loadings_ # shape: (36, 10)
Y_loadings = model_cca.y_loadings_

def plot_CCA(n_keep, grid_n, variate_weights, labels):
    for n in range(n_keep):
        plot = plt.figure(figsize=(10, 7))

        grid = np.zeros((grid_n, grid_n))
    
        triu_mask = np.triu(np.ones_like(grid, dtype=np.bool))
    
        weights = np.tril(variate_weights[:, n])
    
        TH = 0.00
        weights[(weights < TH) & (weights > -TH)] = 0

        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        ax = sns.heatmap(data=weights, mask=triu_mask, cbar=True, linewidths=.5,
                         vmin=-0.5, vmax=0.5, center=0,
                         cmap=cmap, square=True, 
                         cbar_kws={"shrink": .5})
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels, rotation=0)
        plt.title('Canonical component %i in Social Brain subnodes' % (n + 1))
        plt.tight_layout()
        plt.savefig('%s/%s_CCA.png' % (OUT_DIR, (n+1)), dpi=600, transparent=True)
    return plot 

# plot_CCA(3, 36, X_loadings, rois)

sb_columns = ["sb1", "sb2", "sb3", "sb4", "sb5", "sb6", "sb7", "sb8", "sb9", "sb10"]
FSL_columns = ["fsl1", "fsl2", "fsl3", "fsl4", "fsl5", "fsl6", "fsl7", "fsl8", "fsl9", "fsl10"]
sb_cca = pd.DataFrame(model_cca.x_scores_, columns=sb_columns)
fsl_cca = pd.DataFrame(model_cca.y_scores_, columns=FSL_columns)

# finally, concatenate your features!
X_train_features = pd.concat([sb_cca, fsl_cca], axis=1)


# start building models!
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

X_train_features_np = np.array(X_train_features)
y_train_np = np.array(y_train)


# baseline model: Logistic Regression 
folder = KFold(n_splits=10)
est = LogisticRegression(random_state=42)
cv_acc = cross_val_score(est, X_train_features, y_train, cv=folder, verbose=1)
print('Final score: %2.10f%%' % (np.mean(cv_acc) * 100))
# Final score: 72.0500649491%

# plot the confusion matrix of results
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import seaborn as sns

predictions = cross_val_predict(est, X_train_features, y_train, cv=folder)
cm = metrics.confusion_matrix(y_train, predictions)


cmap = sns.diverging_palette(50, 9, n=4, as_cmap=True)

cmap = sns.diverging_palette(9,255, n=4, as_cmap=True)

plot = plt.figure(figsize=(10, 7))
ax =sns.heatmap(cm, square=True, center=0, 
                cmap=cmap, linewidths=4.0,
                linecolor="#FFFFFF", fmt=".3f",
                annot=True)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

plt.savefig('%s/logreg_heatmap.png' % (OUT_DIR), dpi=400, transparent=True)





# model 2: grid search Logistic Regression
from sklearn.model_selection import GridSearchCV

folder = KFold(n_splits=10, shuffle=True)
est = LogisticRegression(random_state=42)

outer_acc_train = []
outer_acc_test = []
for train, test in folder.split(X_train_features_np): # outer CV fold
    print("TRAIN:", train[:5], "TEST:", test[:5])
    X_train_gs, X_test_gs = X_train_features_np[train], X_train_features_np[test]
    y_train_gs ,y_test_gs = y_train_np[train], y_train_np[test]

    my_grid = {
            'penalty' : ['l1', 'l2'],
            'C' : np.linspace(0.1,2,30),
            'solver' : ['liblinear']} 

    folder_inner = KFold(n_splits=5)
    gs_est = GridSearchCV(estimator=est, param_grid=my_grid,
        n_jobs=4, cv=folder_inner, verbose=True)
    gs_est.fit(X_train_gs, y_train_gs)
    print(gs_est.best_params_)

    outer_acc_train.append(gs_est.score(X_train_gs, y_train_gs))
    outer_acc_test.append(gs_est.score(X_test_gs, y_test_gs))

print('Final score: %2.10f%%' % (np.mean(outer_acc_test) * 100))
# Final score: 72.0630367323%



# model 3: Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

folder = KFold(n_splits=10, shuffle=True)
est = RandomForestClassifier(random_state=42)
cv_acc = cross_val_score(est, 
                X_train_features_np, y_train_np, 
                cv=folder, verbose=1)

print('Final score: %2.10f%%' % (np.mean(cv_acc) * 100))
# Final score: 70.0764595511%


# model 4: grid search Random Forest Classifier TOO LONG!!  

folder = KFold(n_splits=10, shuffle=True)
est = RandomForestClassifier(random_state=42)

outer_acc_train = []
outer_acc_test = []
for train, test in folder.split(X_train_features_np): # outer CV fold
    print("TRAIN:", train[:5], "TEST:", test[:5])
    X_train_gs, X_test_gs = X_train_features_np[train], X_train_features_np[test]
    y_train_gs ,y_test_gs = y_train_np[train], y_train_np[test]

    my_grid = {
            'n_estimators' : np.linspace(10,500,50, dtype=int),
            'max_depth' : np.linspace(5,30,6, dtype=int),
            'min_samples_split' : np.linspace(2,100,50, dtype=int),
            'min_samples_leaf': np.linspace(1,10,10, dtype=int)    
            } 

    folder_inner = KFold(n_splits=5)
    gs_est = GridSearchCV(estimator=est, param_grid=my_grid,
        n_jobs=4, cv=folder_inner, verbose=True)
    gs_est.fit(X_train_gs, y_train_gs)
    print(gs_est.best_params_)

    outer_acc_train.append(gs_est.score(X_train_gs, y_train_gs))
    outer_acc_test.append(gs_est.score(X_test_gs, y_test_gs))

print('Final score: %2.10f%%' % (np.mean(outer_acc_test) * 100))
# Final score:



# model 4b: grid search Random Forest Classifier 
from sklearn.ensemble import RandomForestClassifier
folder = KFold(n_splits=10, shuffle=True)
est = RandomForestClassifier(random_state=42)

outer_acc_train = []
outer_acc_test = []
for train, test in folder.split(X_train_features_np): # outer CV fold
    print("TRAIN:", train[:5], "TEST:", test[:5])
    X_train_gs, X_test_gs = X_train_features_np[train], X_train_features_np[test]
    y_train_gs ,y_test_gs = y_train_np[train], y_train_np[test]

    my_grid = {
            'n_estimators' : np.linspace(10,500,5, dtype=int),
            'max_depth' : np.linspace(5,30,6, dtype=int),
            } 

    folder_inner = KFold(n_splits=5)
    gs_est = GridSearchCV(estimator=est, param_grid=my_grid,
        n_jobs=4, cv=folder_inner, verbose=True)
    gs_est.fit(X_train_gs, y_train_gs)
    print(gs_est.best_params_)

    outer_acc_train.append(gs_est.score(X_train_gs, y_train_gs))
    outer_acc_test.append(gs_est.score(X_test_gs, y_test_gs))

print('Final score: %2.10f%%' % (np.mean(outer_acc_test) * 100))
# Final score: 71.9965721296%






# model 5: gradient boosting classifier 
from sklearn.ensemble import GradientBoostingClassifier


folder = KFold(n_splits=10, shuffle=True)
est = GradientBoostingClassifier(random_state=42)
cv_acc = cross_val_score(est, X_train_features_np, 
            y_train_np, cv=folder, verbose=1)

print('Final score: %2.10f%%' % (np.mean(cv_acc) * 100))
# Final score: 71.7408890813%



# model 6: grid search gradient boosting classifer 

folder = KFold(n_splits=10, shuffle=True)
est = GradientBoostingClassifier(random_state=42)

outer_acc_train = []
outer_acc_test = []
for train, test in folder.split(X_train_features_np): # outer CV fold
    print("TRAIN:", train[:5], "TEST:", test[:5])
    X_train_gs, X_test_gs = X_train_features_np[train], X_train_features_np[test]
    y_train_gs ,y_test_gs = y_train_np[train], y_train_np[test]

    my_grid = {
            'learning_rate' : [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1],
            'n_estimators' : np.linspace(10,500,20, dtype=int),
            } 

    folder_inner = KFold(n_splits=5)
    gs_est = GridSearchCV(estimator=est, param_grid=my_grid,
        n_jobs=2, cv=folder_inner, verbose=True)
    gs_est.fit(X_train_gs, y_train_gs)
    print(gs_est.best_params_)

    outer_acc_train.append(gs_est.score(X_train_gs, y_train_gs))
    outer_acc_test.append(gs_est.score(X_test_gs, y_test_gs))

print('Final score: %2.10f%%' % (np.mean(outer_acc_test) * 100))
# Final score: 71.9020350725%



# model 7: XGBoost 
from xgboost import XGBClassifier
folder = KFold(n_splits=10, shuffle=True)
est = XGBClassifier(random_state=42)


cv_acc = cross_val_score(est, X_train_features_np, y_train_np, cv=folder, verbose=1)
print('Final score: %2.10f%%' % (np.mean(cv_acc) * 100))