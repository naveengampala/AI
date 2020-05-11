import config 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

#Day2 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class Day:
        
    def read_csv():
        """
            Reading a Dataframe using pandas
        """
        data = pd.read_csv(config.TRAINING_FILE)
        return data
    
    def drop_features(data):
        drop_df = data.drop(['Name', 'Cabin', 'Ticket', 'Sex', 'SibSp', 'Parch', 'Embarked'], axis = 1) 
        return drop_df

    def handle_missing_values(data):
        """
            Handles missing values and returns total and percentage 
        """
        total = data.isnull().sum().sort_values(ascending = False)
        percentage = round(total / data.shape[0] * 100)
        return pd.concat([total, percentage], axis = 1, keys = ['total', 'percentage'])

    def train_and_test(encoded_data):
        """
            Spliting Dataset into Trian and Test using sklearn
        """
        encoded_data= encoded_data.dropna()
        X = encoded_data.loc[:,encoded_data.columns != 'Survived'] 
        y = encoded_data.Survived
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
        return X_train, X_test, y_train, y_test     
    
    def model_train_and_predict(X_train, X_test, y_train, y_test):
        """
            Simple Logistic Regression Model to the Training Set
        """
        model = LogisticRegression()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
        return accuracy, prediction

    # Need to Implement Visualization
    def visualization (X_train, X_test):
        plt.scatter(X_train, X_test)
        plt.show()


if __name__ == "__main__" :
    data = Day.read_csv()
    misisng_percentage = Day.handle_missing_values(data)
    drop_df = Day.drop_features(data)
    X_train, X_test, y_train, y_test = Day.train_and_test(drop_df)
    accuracy, prediction = Day.model_train_and_predict(X_train, X_test, y_train, y_test)
    # Day.visualization (X_train, X_test)