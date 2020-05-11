import config 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

#Day2 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class Day:
        
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

    def train_and_test(data):
        """
            Spliting Dataset into Trian and Test using sklearn
        """
        data.fillna(0, axis=1, inplace=True)
        #Selecting features
        features = [
            'actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross','num_voted_users',
            'cast_total_facebook_likes', 'facenumber_in_poster', 'num_user_for_reviews', 'budget',
            'title_year','actor_2_facebook_likes', 'aspect_ratio', 'movie_facebook_likes'
        ]
        target = ['imdb_score']
        X = data[features].dropna() 
        y = data[target].dropna()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
        return X_train, X_test, y_train, y_test     
    
    def model_train_and_predict(X_train, X_test, y_train, y_test):
        """
            Simple Linear Regression Model to the Training Set
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        error = mean_squared_error(y_test, prediction)
        return error, prediction


    # Need to Implement Visualization
    def visualization (X_train, X_test):
        plt.scatter(X_train, X_test)
        plt.show()


if __name__ == "__main__" :
    data = Day.read_csv()
    misisng_percentage = Day.handle_missing_values(data)
    X_train, X_test, y_train, y_test = Day.train_and_test(data)
    error, prediction = Day.model_train_and_predict(X_train, X_test, y_train, y_test)
    print (error)
    # Day.visualization (X_train, X_test)