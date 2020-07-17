'''
Created on 6/21/20

@author: dulanj
'''
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from DataLoader import DataLoader, Columns
import numpy as np
from sklearn import metrics
from Submission import Submission
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint


class FairClassification(object):
    def __init__(self):
        self.data_loader = DataLoader()
        self.data_loader.clean_data()
        self.data_loader.surge_or_not()
        self.train_df, self.test_df = self.data_loader.get_dataframes()
        self.submit = Submission('random_forest_hpt.csv')

    def hyper_parameter_tuning(self):
        y = self.train_df[Columns.label]
        X = self.train_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time, Columns.label], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)
        train_features = X_train
        train_labels = y_train
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]
        # Number of features to consider at every split
        max_features = [None, 'auto', 'log2', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]  # Create the random grid
        n_jobs = [-1]
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap,
                       'n_jobs': n_jobs}
        pprint(random_grid)

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                       random_state=42, n_jobs=-1)  # Fit the random search model
        rf_random.fit(train_features, train_labels)

        print(rf_random.best_params_)

    def random_forest(self):
        y = self.train_df[Columns.label]
        X = self.train_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time, Columns.label], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        random_grid = {'n_estimators': 733,
                       'max_features': None,
                       'max_depth': 110,
                       'min_samples_split': 5,
                       'min_samples_leaf': 1,
                       'bootstrap': True}
        regressor = RandomForestClassifier(n_estimators=random_grid['n_estimators'],
                                           max_features=random_grid['max_features'],
                                           max_depth=random_grid['max_depth'],
                                           min_samples_split=random_grid['min_samples_split'],
                                           min_samples_leaf=random_grid['min_samples_leaf'],
                                           bootstrap=random_grid['bootstrap'],
                                           random_state=0)

        # regressor = RandomForestClassifier(n_estimators=500, random_state=0)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        print("Accuracy {}: {}".format(500, metrics.accuracy_score(y_test, y_pred)))

        # predict_df = self.test_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time], axis=1)
        # length_of_test = (predict_df.iloc[:, 1].count())
        # print("length of test: {}".format(length_of_test))
        # for i in range(length_of_test):
        #     print(i)
        #     input_data = predict_df.iloc[i].values
        #     input_data_dim = np.expand_dims(input_data, axis=0)
        #     p = regressor.predict(input_data_dim)
        #     self.submit.write(self.test_df[Columns.trip_id][i], p[0])

    def decision_trees(self):
        y = self.train_df[Columns.label]
        X = self.train_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time, Columns.label], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
            clf.score(X_test, np.ravel(y_test, order='C'))))

        predict_df = self.test_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time], axis=1)
        length_of_test = (predict_df.iloc[:, 1].count())
        print("length of test: {}".format(length_of_test))
        for i in range(length_of_test):
            input_data = predict_df.iloc[i].values
            input_data_dim = np.expand_dims(input_data, axis=0)
            p = clf.predict(input_data_dim)
            self.submit.write(self.test_df[Columns.trip_id][i], p[0])


if __name__ == "__main__":
    obj = FairClassification()
    obj.random_forest()