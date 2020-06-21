'''
Created on 6/21/20

@author: dulanj
'''
import pandas as pd
import logging


class DataLoader(object):
    def __init__(self, train_filename, test_filename):
        logging.basicConfig(level=logging.INFO)
        self.train_filename = train_filename
        self.test_filename = test_filename

        self.train_df = pd.read_csv(self.train_filename)
        logging.info("Train DF loaded\n{}".format(self.train_df.head()))
        self.test_df = pd.read_csv(self.test_filename)
        logging.info("Test DF loaded\n{}".format(self.test_df.head()))

    def get_dataframes(self):
        return self.train_df, self.test_df


if __name__ == "__main__":
    train_filename = 'data/train.csv'
    test_filename = 'data/test.csv'
    obj = DataLoader(train_filename, test_filename)
    obj.get_dataframes()