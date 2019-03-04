from dataReader import dataReader
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
import pandas as pd


class PredictClick():
    def trainModel(self, xTrain, yTrain):
        print("Train")
        xTrain = xTrain.apply(pd.to_numeric, errors='coerce')
        yTrain = yTrain.apply(pd.to_numeric, errors='coerce')
        xTrain.fillna(0, inplace=True)
        yTrain.fillna(0, inplace=True)

        # self._model = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=0, n_jobs=2)
        self._model = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=0)

        self._model = self._model.fit(xTrain, yTrain)

    def validateModel(self, xValidation, yValidation):
        print("Validation")
        xValidation = xValidation.apply(pd.to_numeric, errors='coerce')
        yValidation = yValidation.apply(pd.to_numeric, errors='coerce')
        xValidation.fillna(0, inplace=True)
        yValidation.fillna(0, inplace=True)

        pred = self._model.predict_proba(xValidation)
        pred = pred[:, 1]

        predDF = pd.DataFrame(pred, columns=["click"]).to_csv('pClick.csv')
        R_Square = r2_score(yValidation, pred)

        print(R_Square)

if __name__ == "__main__":
    trainset="../we_data/train.csv"
    validationset="../we_data/validation.csv"
    testset="../we_data/test.csv"

    trainReader = dataReader(trainset)
    validationReader = dataReader(validationset)

    trainDF = trainReader.getDataFrame()
    validationDF = validationReader.getDataFrame()

    xTrain = trainReader.getOneHotData()
    yTrain = trainReader.getDataFrame()['click']
    xValidation = validationReader.getOneHotData()
    yValidation = validationReader.getDataFrame()['click']

    pc = PredictClick()
    pc.trainModel(xTrain, yTrain)
    pc.validateModel(xValidation, yValidation)
    print('...')
