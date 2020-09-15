'''
Created on 6/21/20

@author: dulanj
'''
from pprint import pprint

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from DataLoader import DataLoader, Columns
from Submission import Submission


class FairClassification(object):
    def __init__(self):
        self.data_loader = DataLoader()
        # self.data_loader.clean_data()
        self.data_loader.surge_or_not()
        self.data_loader.straight_distance()
        self.data_loader.geo_location()
        self.train_df, self.test_df = self.data_loader.get_dataframes()
        self.submit = None

    def hyper_parameter_tuning(self):
        """":arg
        Hyperparameter tunining for the Random forest algorithm
        """
        y = self.train_df[Columns.label]
        X = self.train_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time, Columns.label], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)
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
        """:arg
        Random forest algorithm
        """
        self.submit = Submission('random_forest_with_distance_surge_geo-location.csv')
        y = self.train_df[Columns.label]
        X = self.train_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time, Columns.label], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)

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
        print("Training...")
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        print("Accuracy {}: {}".format(500, metrics.accuracy_score(y_test, y_pred)))

        predict_df = self.test_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time], axis=1)
        length_of_test = (predict_df.iloc[:, 1].count())
        print("length of test: {}".format(length_of_test))
        for i in range(length_of_test):
            print(i)
            input_data = predict_df.iloc[i].values
            input_data_dim = np.expand_dims(input_data, axis=0)
            p = regressor.predict(input_data_dim)
            self.submit.write(self.test_df[Columns.trip_id][i], p[0])

    def decision_trees(self):
        """:arg
        Decision trees algorithm
        """
        self.submit = Submission('decision_trees.csv')
        y = self.train_df[Columns.label]
        X = self.train_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time, Columns.label], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
            clf.score(X_test, np.ravel(y_test, order='C'))))

        predict_df = self.test_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time], axis=1)
        length_of_test = (predict_df.iloc[:, 1].count())
        print(predict_df.columns)
        print("length of test: {}".format(length_of_test))
        for i in tqdm(range(length_of_test)):
            # print(i)
            input_data = predict_df.iloc[i].values
            input_data_dim = np.expand_dims(input_data, axis=0)
            # print(input_data_dim)
            p = clf.predict(input_data_dim)
            self.submit.write(self.test_df[Columns.trip_id][i], p[0])

    def knn(self):
        from sklearn.neighbors import KNeighborsClassifier
        self.submit = Submission('knn.csv')
        y = self.train_df[Columns.label]
        X = self.train_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time, Columns.label], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)

        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
            clf.score(X_test, np.ravel(y_test, order='C'))))

        predict_df = self.test_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time], axis=1)
        length_of_test = (predict_df.iloc[:, 1].count())
        print(predict_df.columns)
        print("length of test: {}".format(length_of_test))
        for i in tqdm(range(length_of_test)):
            # print(i)
            input_data = predict_df.iloc[i].values
            input_data_dim = np.expand_dims(input_data, axis=0)
            # print(input_data_dim)
            p = clf.predict(input_data_dim)
            self.submit.write(self.test_df[Columns.trip_id][i], p[0])

    def svm(self):
        from sklearn import svm
        self.submit = Submission('svm.csv')
        y = self.train_df[Columns.label]
        X = self.train_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time, Columns.label], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)

        clf = svm.SVC(kernel='linear')
        for i in range(int(len(X_train)/100)-1):
            print("{} / {}".format(i, int(len(X_train)/100)-1))
            clf.partial_fit(X_train[i*100:(i+1)*100], y_train[i*100:(i+1)*100])
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
            clf.score(X_test, np.ravel(y_test, order='C'))))

        predict_df = self.test_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time], axis=1)
        length_of_test = (predict_df.iloc[:, 1].count())
        print(predict_df.columns)
        print("length of test: {}".format(length_of_test))
        for i in tqdm(range(length_of_test)):
            # print(i)
            input_data = predict_df.iloc[i].values
            input_data_dim = np.expand_dims(input_data, axis=0)
            # print(input_data_dim)
            p = clf.predict(input_data_dim)
            self.submit.write(self.test_df[Columns.trip_id][i], p[0])

    def ann(self):
        from sklearn import svm
        self.submit = Submission('ann.csv')
        y = self.train_df[Columns.label]
        X = self.train_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time, Columns.label], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)

        import tensorflow as tf
        # define the keras model
        model = tf.keras.Sequential()
        # model.add(tf.keras.layers.Dense(841, input_dim=8, activation='relu'))
        model.add(tf.keras.layers.Dense(256,  input_dim=838,activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(8, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=1)

        predict_df = self.test_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time], axis=1)
        length_of_test = (predict_df.iloc[:, 1].count())
        print(predict_df.columns)
        print("length of test: {}".format(length_of_test))
        for i in tqdm(range(length_of_test)):
            # print(i)
            input_data = predict_df.iloc[i].values
            input_data_dim = np.expand_dims(input_data, axis=0)
            # print(input_data_dim)
            p = model.predict(input_data_dim)
            self.submit.write(self.test_df[Columns.trip_id][i], p[0])


if __name__ == "__main__":
    obj = FairClassification()
    # obj.decision_trees()
    # obj.hyper_parameter_tuning()
    obj.svm()
