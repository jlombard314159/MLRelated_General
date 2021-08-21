from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,\
     RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
     BaggingClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pylab as plt
from subprocess import call
from sklearn.tree import export_graphviz 
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=5, n_jobs=8, verbose=1, refit=False, scoring=None):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, refit=refit,
                              return_train_score=True, scoring = scoring)
            gs.fit(X,y)
            self.grid_searches[key] = gs

        self.response = y
        self.covariates = X

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

        self.topModel = self.extractTopScoringModel(df[columns])

        return df[columns]

    def extractTopScoringModel(self,df):

        scoreSummary = df.iloc[0,]

        scoreSummary = scoreSummary.dropna()

        return scoreSummary

models_regression = {'XGB': XGBRegressor(),
    'RandForest': RandomForestRegressor(),
    'GradBoost': GradientBoostingRegressor()}

params_regression = {
'XGB': {'max_depth':[3,6,12,20,24],
            'learning_rate':[0.001,0.05,0.1,1],
            'gamma':[0,0.001,0.01,0.1,0.5]},
'RandForest': {'max_depth':[3,6,12,20],
            'n_estimators': [2,4,8,16,32]},
'GradBoost': {'n_estimators': [2,4,8,16,32],
                                    'learning_rate': [0.001,0.05,0.5,1]}
}


models_classifier = {
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'AdaBoostClassifier': AdaBoostClassifier(random_state=42),
    'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
    'BaggingClassifier': BaggingClassifier(random_state=42),
    'XGB': XGBClassifier(use_label_encoder=False),
    'LogReg':LogisticRegression(random_state=42, solver ='liblinear'),
    'ElasticNet':LogisticRegression(random_state=42, solver='saga')
}

params_classifier = {
    'RandomForestClassifier': {'n_estimators': [2,4,8,16,32,128,500,1000],
                               'max_depth': [3,6,12,20]},
    'AdaBoostClassifier':  { 'n_estimators': [2,4,8,16, 32,128,500,1000],
                             'learning_rate': [0.001,0.05,0.5,1.0,3]},

    'GradientBoostingClassifier': { 'n_estimators': [2,4,8,16, 32,128,500, 1000],
                                    'learning_rate': [0.001,0.05,0.5,1]},

    'BaggingClassifier': {'n_estimators': [2,4,8,16,32,128, 500, 1000]},

    'XGB': {'max_depth':[3,6,12,20],
            'learning_rate':[0.001,0.05,0.1,1],
            'gamma':[0,0.001,0.01,0.1,0.5]},

    'LogReg':{'penalty':['l1','l2'],
              'C': [0.0001,0.001,0.01,0.1,0.5,1,2,10]},

    'ElasticNet':{'penalty':['elasticnet'],
              'l1_ratio':[0,0.0001,0.001,0.01,0.1,0.5,1]}

}

def createEstimatorSelection(models,params):

    estimateGrid = EstimatorSelectionHelper(models, params)

    return estimateGrid

def fitEstimatorSelection(estimatorSelect, covariates, response, scoring,label):

    resultsDict = {}
    for label, data in zip(label,response):
        estimatorSelect.fit(X=covariates, y=data,scoring = scoring)

        top, all = extractTopInfo(estimatorSelect)
        resultsDict[label] = {'topScore':top,'allScores':all}

    return resultsDict

def shuffleData(covariates,response):

    covariates = covariates.sample(frac=1, random_state=42)

    newIndex = covariates.index

    shuffledDict = {}
    for key, data in response.items():
        newData = data.reindex(newIndex)
        shuffledDict[key] =newData

    return covariates, shuffledDict

def convertToClassification(response):

    binaryDict = {}
    for key, data in response.items():
        data.iloc[data <= 0] = 0
        data.iloc[data > 0] = 1
        data = data.astype('int32')
        binaryDict[key] =data

    return binaryDict

def extractTopInfo(estimatorObj):

    topSummary = estimatorObj.score_summary()
 
    topData = estimatorObj.topModel

    return topData, topSummary

def featureImportance(model, columnNames):

    forest_importances = pd.Series(model.feature_importances_, index=columnNames)

    forestDF = pd.DataFrame({'covariate':forest_importances.index, 'importances':forest_importances.values})

    return forestDF

def selectHighestImportances(importanceDF, topN = -0):

    valToQuery = sorted(importanceDF['importances'])[-topN]

    highestDF = importanceDF[importanceDF['importances'] > valToQuery]

    return highestDF

def createTreePlot(oneTree, columnLabels, classNames, fileName):

    export_graphviz(oneTree, 
                    out_file= fileName + '.dot', 
                    feature_names = columnLabels,
                    class_names = classNames,
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)

    call(['dot', '-Tpng', fileName + '.dot', '-o', fileName +'.png', '-Gdpi=600'])


    return None

def importancePlot(importanceDF, savePath, modelAppender):
 
    ax = plt.gca()
    ax.bar(importanceDF['covariate'],importanceDF['importances'])

    plt.xlabel('Covariate')
    plt.ylabel('Mean accuracy decrease')
    plt.title('Top feature importances using permutation on full model' + modelAppender)
    plt.xticks(rotation =45, fontsize = 5)
    plt.axis('tight')
    plt.savefig(savePath + "/" + modelAppender + "_ImportancePlot.png", dpi=140)
    plt.clf()

    return None

def addPredictions(model, modelingData,responseData):

    predictions = model.predict(modelingData)

    outputDF = pd.DataFrame()
    outputDF['actual'] = responseData

    outputDF['prediction'] = predictions

    return outputDF

def modelAccuracies(outputDF):


    trueNeg, falsePos, falseNeg, truePos = confusion_matrix(outputDF['actual'],
        outputDF['prediction']).ravel()

    accuracy = (trueNeg + truePos)/(truePos + trueNeg + falseNeg + falsePos)

    precision = truePos/(truePos + falsePos)
    recall = truePos/(truePos + falseNeg)

    f1_score = 2 * ((precision*recall)/(precision + recall))

    dataToInsert = {'true negatives':[trueNeg],
        'false positives': [falsePos],
        'false negatives': [falseNeg],
        'true positive': [truePos]}

    prettyDF = pd.DataFrame(dataToInsert)

    outputDict = {'confusion matrix': prettyDF, 'accuracy':accuracy,
        'f1':f1_score}

    return outputDict

def baggingFeatureImportance(baggingModel, covarNames):

    feature_importances = np.mean([tree.feature_importances_ for tree in 
        baggingModel.estimators_], axis=0)

    forestDF = pd.DataFrame({'covariate':covarNames, 'importances':feature_importances})

    return forestDF

def createSensitivityData(covariateData, columnToVary):

    valuesToUse = covariateData[columnToVary]

    scale = abs(valuesToUse.mean()/50)

    covarSeq = np.arange(valuesToUse.min(),valuesToUse.max(),scale)

    restCovariates = covariateData.drop(columnToVary, 1)

    avgCovariates = restCovariates.min(axis=0)

    finalDF = pd.DataFrame(index = list(range(0,len(covarSeq),1)), columns = covariateData.columns)
    finalDF = finalDF.fillna(0)

    for index in range(0,len(covarSeq),1):
        finalDF.iloc[index] = [covarSeq[index]] + list(avgCovariates)

    return finalDF

def modelSensitivityData(sensitivityData, model):

    predictionList = []

    for _, row in sensitivityData.iterrows():

        rowPrediction = model.predict(row.to_numpy()[np.newaxis])
        predictionList.append(rowPrediction)

    flattenedList = [item for sublist in predictionList for item in sublist]

    updatedDF = sensitivityData
    updatedDF['Prediction'] = flattenedList

    return updatedDF
    

def calculateBreakPoint(predictionSensitivData, columnToPick):

    vectorValue = predictionSensitivData['Prediction'].to_numpy()

    indexForChange = np.where(vectorValue[:-1] != vectorValue[1:])[0]
    previousIndex = indexForChange - 1

    if indexForChange.size == 0:
        changeAndBeforeChangeValues = predictionSensitivData[columnToPick].iloc[0]
        constantPrediction = 'Yes'

    else:
        changeAndBeforeChangeValues = predictionSensitivData[columnToPick].iloc[previousIndex:(indexForChange+1)]
        constantPrediction = 'No'

    constantPredID = 'constantPred' + columnToPick

    returnDict = {columnToPick: changeAndBeforeChangeValues,
        constantPredID : constantPrediction}

    return returnDict
