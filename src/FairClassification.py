'''
Created on 6/21/20

@author: dulanj
'''
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from DataLoader import DataLoader, Columns
import numpy as np

from Submission import Submission


class FairClassification():
    def __init__(self):
        self.data_loader = DataLoader()
        self.data_loader.clean_data()
        self.train_df, self.test_df = self.data_loader.get_dataframes()
        self.submit = Submission('decision_trees.csv')

    def decision_trees(self):
        y = self.train_df[Columns.label]
        X = self.train_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time, Columns.label], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)

        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        # print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
        #     clf.score(X_test, np.ravel(y_test, order='C'))))

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
    obj.decision_trees()