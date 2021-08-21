import pandas as pd
import numpy as np

#Set seed
np.random.seed(1)

#-----------------------------------------------------------------
#Data is loaded and split correctly now

#X is the train
#y is label for train

#X_test/y_test are our results

##Keep the models with no transformations for now

class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=10, n_jobs=8, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]

#----------------------------------------------------------------
##Grid search for arbitrary estimator
#SVM
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, \
    AdaBoostClassifier, GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

from xgboost import XGBClassifier



# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('clf', EstimatorSelectionHelper())
# ])
#--------------------------------------------------------------------------------
#Class wieight sucks try resampling based on the lower class

from random import sample

def Undersample_DF(dataDF,labelDF,classes = [0,1]):

    # pdb.set_trace()
    #First find the bigger class
    numberFirstClass = (labelDF == classes[0]).sum()
    numberSecondClass = (labelDF == classes[1]).sum()

    #Pick the index to use
    if numberFirstClass < numberSecondClass:
        classToUse = classes[0]
        classToSample = classes[1]
    else:
        classToUse = classes[1]
        classToSample = classes[0]

    undersampleIndex = [i for i, e in enumerate(labelDF) if e == classToUse]
    removeIndex = [i for i, e in enumerate(labelDF) if e == classToSample]

    #Store what we are not sampling from
    dataDF_Store = dataDF[undersampleIndex,:]

    removeIndexSample = sample(removeIndex,len(undersampleIndex))
    dataDF_Sample = dataDF[removeIndexSample,:]

    newDataDF = np.concatenate([dataDF_Store,dataDF_Sample])
    newLabelDF = np.concatenate([labelDF[undersampleIndex],
                           labelDF[removeIndexSample]])

    return newDataDF, newLabelDF

X_train_final, y_train_final = Undersample_DF(dataDF = train_X,
                                              labelDF = train_Y)

#---------------------------------------------------------------------------------
#Oversample dataset
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 33)

X_train_final_oversample, y_train_oversample = sm.fit_resample(train_X, train_Y)

#---------------------------------------------------------------------------------

classWeight = dict({0:1, 1:10})
models1 = {
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'BaggingClassifier': BaggingClassifier(),
    'SVC': SVC(),
    'kNN': KNeighborsClassifier(),
    'SGDClassifier': SGDClassifier(),
    'XGB': XGBClassifier()
}

params1 = {
    'RandomForestClassifier': {'n_estimators': [8,16,32,128],
                               'max_depth': [3,6,12,20]},
    'AdaBoostClassifier':  { 'n_estimators': [16, 32,128],
                             'learning_rate': [0.001,0.05,0.5,1.0,3]},
    'GradientBoostingClassifier': { 'n_estimators': [8,16, 32,128],
                                    'learning_rate': [0.001,0.05,0.5,1]},
    'BaggingClassifier': {'n_estimators': [8,16,32,128]},
    'SVC': [
        {'kernel': ['rbf'], 'C': [1, 10, 50], 'gamma': [0.001, 0.0001]},
        {'kernel': ['poly'], 'coef0': [0.5, 1, 5], 'degree':[2, 3, 5], 'C':[1, 10, 50]}
    ],
    'kNN':[
        {'n_neighbors':[8,20,40,128],'weights':['uniform','distance']}
    ],
    'SGDClassifier': {'loss': ['log','hinge'], 'penalty': ['l1', 'elasticnet'],
                     'l1_ratio': [0.1,0.5,0.7,0.9,0.95,0.99,1], 'alpha': [0.0001,0.01,0.5,1]},
    'XGB': {'max_depth':[3,6,12,20],
            'learning_rate':[0.001,0.05,0.1,1],
            'gamma':[0,0.001,0.01,0.1,0.5]}
}

helper1 = EstimatorSelectionHelper(models1, params1)
helper1.fit(X_train_final_oversample, y_train_oversample, scoring='f1', n_jobs=7)

finalOutput1 = helper1.score_summary()

finalOutput1.to_csv('LACI_Species_PHKC_Parameters.csv',sep=",",index=False)
#---------------------------------------------------------------------------
#All said and done then just read in the csv

finalOutput1 = pd.read_csv('All_Species_PHKC_Parameters.csv')


#Try and get the top two methods that are different
firstEstimator = finalOutput1.iloc[0]
firstEstimator = firstEstimator.dropna()

secondEstimator = finalOutput1[finalOutput1.estimator != firstEstimator.estimator]
secondEstimator = secondEstimator.iloc[0]
secondEstimator = secondEstimator.dropna()

topModelPredict = RandomForestClassifier(max_depth=20,
                                         n_estimators=128)

# secondModelPredict = XGBClassifier(max_depth=6,gamma=0.01,learning_rate=0.1,probability=True)
secondModelPredict = SVC(C=50,kernel='rbf',gamma=0.001,probability=True)

# ##For fats
# topModelPredict = SVC(C=60,gamma=0.001,kernel='rbf',probability=True)
# secondModelPredict = SGDClassifier(alpha = 0.001, l1_ratio = 0.5,
#                                    loss = 'log', penalty= 'elasticnet')

topModelPredict.fit(X_train_final_oversample,y=y_train_oversample)
secondModelPredict.fit(X_train_final_oversample,y=y_train_oversample)

# #----------------------------------------------------------------------------
# #Manual stuff below
#
# #Now look at FP/TP
# #This first model is when I use all training data
# ##Note that I changed loss from hinge to log
# tempFit = SGDClassifier(alpha=0.01, l1_ratio=0.95,
#                         loss='hinge', penalty='elasticnet')
#
# tempFitFinal = tempFit.fit(X_train,y)
#
# from sklearn.calibration import CalibratedClassifierCV
# calibrator = CalibratedClassifierCV(tempFit, cv='prefit')
# modelFinal = calibrator.fit(X_train,y)
#
#
# # tempPred = tempFit.predict(X_test)
# #
# # precision_score(y_test,tempPred)
# # recall_score(y_test,tempPred)
#
# #Now look at the best for the undersampled data
# #TOp model below
# tempFit = RandomForestClassifier(max_depth=10, n_estimators=32)
# tempFit.fit(X_train_final,y_train_final)
#
# modelFinal_2 = tempFit.fit(X_train_final,y_train_final)
#
# #Now look at oversampling
# #Top model below
# tempFit = RandomForestClassifier(max_depth=10,n_estimators=128)
# model3 = tempFit.fit(X_train_final_oversample,y_train_oversample)

# #-----------------------------------------------------------
# #For fats do jacknife
# dataVecOne = []
#
# for i in range(train_X.__len__()):
#
#     jackKnifeData = train_X[i]
#
#     predToStore = topModelPredict.predict_proba(jackKnifeData.reshape(1,-1))[:]
#     predToStore = predToStore.ravel()
#
#     dataVecOne.append(predToStore[0])
#
#
# dataVecOne = [0 if i >=0.5 else 1 for i in dataVecOne]


#----------------------------------------------------------
#Make plots
dataSetPath = 'C:/Users/jlombardi/Documents/GitLabCode/nnplay/Classifier Pipeline/Data/PHKH_ROC.csv'

dataset = pd.read_csv(dataSetPath)

dataset = dataset[dataset.species == 'All']


from sklearn.metrics import roc_curve, auc
# Compute ROC curve and ROC area for each class

y_pred_rf = topModelPredict.predict_proba(valid_X)[:,1]
y_pred_second = secondModelPredict.predict_proba(valid_X)[:,1]

fpr_rf, tpr_rf, _ = roc_curve(valid_Y, y_pred_rf)

fpr_second, tpr_second, _ = roc_curve(valid_Y, y_pred_second)

import matplotlib.pyplot as plt

plt.figure(0)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.plot(fpr_second,tpr_second, label='SVC')
plt.plot(dataset.falsePositive, dataset.truePositive, label='Elastic Net')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc=0)
plt.show()


#index to replicate beccas graph
firstIndex = [i for i,v in enumerate(tpr_rf) if v >=0.77][0]

tpr_rf[firstIndex]
fpr_rf[firstIndex]


feat_importances = pd.Series(topModelPredict.feature_importances_, index=train_X.columns)
feat_importances.nlargest(8).plot(kind='barh')


#Some broken plotting code
from textwrap import wrap
labels=train_X.columns
labels = [ '\n'.join(wrap(l, 20)) for l in labels ]

iterateMe = feat_importances.nlargest(8)

plt.figure(1)

for i in range(0,8):
    plt.barh(labels[i], iterateMe[i].values)


plt.set_yticklabels(labels)
plt.title('Relative Importance of covariates')
plt.show()