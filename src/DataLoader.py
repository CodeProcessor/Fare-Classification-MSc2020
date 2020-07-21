'''
Created on 6/21/20

@author: dulanj
'''
import pandas as pd
import logging


class Columns:
    """
    Colomns of the dataset
    """
    trip_id = 'tripid'
    additional_fare = 'additional_fare'
    duration = 'duration'
    meter_waiting = 'meter_waiting'
    meter_waiting_fare = 'meter_waiting_fare'
    meter_waiting_till_pickup = 'meter_waiting_till_pickup'
    pickup_time = 'pickup_time'
    drop_time = 'drop_time'
    pick_lat = 'pick_lat'
    pick_lon = 'pick_lon'
    drop_lat = 'drop_lat'
    drop_lon = 'drop_lon'
    fare = 'fare'
    label = 'label'

    surge = 'surge'


class DataLoader(object):
    train_filename = 'data/train.csv'
    test_filename = 'data/test.csv'

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.train_filename = DataLoader.train_filename
        self.test_filename = DataLoader.test_filename

        self.train_df = pd.read_csv(self.train_filename)
        self.test_df = pd.read_csv(self.test_filename)

    def get_dataframes(self):
        """
        Load dataframe
        """
        self.test_df = self.test_df.fillna(0)
        self.train_df = self.train_df.fillna(0)
        logging.info("Test DF loaded\n{}\n".format(self.test_df.head()))
        logging.info("Train DF loaded\n{}\n".format(self.train_df.head()))
        print(self.train_df.columns)
        print(self.test_df.columns)
        return self.train_df, self.test_df

    def clean_data(self):
        """:arg
        Cleaning the data
        Drop NULL values
        """
        logging.info("Length of data: {}".format(len(self.train_df[Columns.trip_id].values)))
        logging.info("Dropping null rows")
        self.train_df.dropna(inplace=True)
        logging.info("Length of data: {}".format(len(self.train_df[Columns.trip_id].values)))

    def straight_distance(self):
        """:arg
        Taking the distance between pickup and drop locations
        """
        def get_dist(row):
            dist = ((row[Columns.pick_lat] - row[Columns.drop_lat])**2 + (row[Columns.pick_lon] - row[Columns.drop_lon])**2)
            return dist

        self.train_df['dist'] = self.train_df.apply(lambda row: get_dist(row), axis=1)
        self.test_df['dist'] = self.test_df.apply(lambda row: get_dist(row), axis=1)
        # self.train_df = pd.concat([self.train_df, train_surge])
        # self.test_df = pd.concat([self.train_df, test_surge])

    def surge_or_not(self):
        """:arg
        Take the pickup hour to get the surge time information
        """
        def get_rejects_percentage(row):
            hour = int(row[Columns.pickup_time].split(' ')[1].split(':')[0])
            return hour
            # return 1 if 17 < hour < 21 or 7 < hour < 10 else 0

        train_surge = self.train_df.apply(lambda row: get_rejects_percentage(row), axis=1)
        test_surge = self.test_df.apply(lambda row: get_rejects_percentage(row), axis=1)
        self.train_df = pd.concat([self.train_df, pd.get_dummies(train_surge, prefix='surge')], axis=1)
        self.test_df = pd.concat([self.test_df, pd.get_dummies(test_surge, prefix='surge')], axis=1)


if __name__ == "__main__":
    obj = DataLoader()
    obj.surge_or_not()
    print(obj.train_df.iloc[41])