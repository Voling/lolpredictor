import fetchData
import io
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from joblib import dump, load





# Use Scikit to create a machine learning model that predicts the compatibility between 5 players using match history data in fetchData.py and 5 players' ranks in fetchData.py

class Model:
    def __init__(self):
        self.accountList = []
        self.model = None
        self.platform = 'NA1'
    def readAccountList(self):
        with open('accountList.txt', 'r') as f:
            self.accountList = f.readlines()
    def train(self):
        for account in self.accountList:
            fetchData.getMatchHistoryData(account, self.platform)
            fetchData.getRank(account, self.platform)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        #train
        model = LinearRegression()
        model.fit(X_train, y_train)

        dump(model, 'model.joblib')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    def eval(self):
        X_train, X_test, y_train, y_test = train.X_train, train.X_test, train.y_train, train.y_test
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"Root Mean Squared Error: {rmse}")




