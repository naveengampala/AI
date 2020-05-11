import config 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

class Day1:
    
    def read_csv():
        """
            Reading a Dataframe using pandas
        """
        data = pd.read_csv(config.TRAINING_FILE)
        return data
    
    def handle_missing_values(data):
        """
            Handles missing values and returns total and percentage 
        """
        total = data.isnull().sum().sort_values(ascending = False)
        percentage = round(total / data.shape[0] * 100)
        return pd.concat([total, percentage], axis = 1, keys = ['total', 'percentage'])

    def encoding_categorical_data(data):
        """
            First Check the what feature are Categorical
            Encode useing get_dummies pandas function
        """
        drop_data = data.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
        encoded_data = pd.get_dummies(drop_data, columns=['Sex', 'Embarked'],drop_first=True)
        return encoded_data

    def train_and_test(encoded_data):
        """
            Spliting Dataset into Trian and Test using sklearn
        """
        X = encoded_data.loc[:,encoded_data.columns != 'Survived'] 
        y = encoded_data.Survived
        X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
        print (X_train.head())
        return X_train, y_train, X_test, y_test     
    
    # Feature Scaling
    


if __name__ == "__main__" :
    data = Day1.read_csv()
    misisng_percentage = Day1.handle_missing_values(data)
    encoded_data =  Day1.encoding_categorical_data(data)
    X_train, y_train, X_test, y_test = Day1.train_and_test(encoded_data)

