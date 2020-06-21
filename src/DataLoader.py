'''
Created on 6/21/20

@author: dulanj
'''
import pandas as pd
import logging

class Columns:
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


class DataLoader(object):
    train_filename = 'data/train.csv'
    test_filename = 'data/test.csv'

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.train_filename = DataLoader.train_filename
        self.test_filename = DataLoader.test_filename

        self.train_df = pd.read_csv(self.train_filename)
        logging.info("Train DF loaded\n{}\n".format(self.train_df.head()))
        self.test_df = pd.read_csv(self.test_filename)
        logging.info("Test DF loaded\n{}\n".format(self.test_df.head()))

    def get_dataframes(self):
        return self.train_df, self.test_df

    def clean_data(self):
        logging.info("Length of data: {}".format(len(self.train_df[Columns.trip_id].values)))
        logging.info("Dropping null rows")
        self.train_df.dropna(inplace=True)
        logging.info("Length of data: {}".format(len(self.train_df[Columns.trip_id].values)))


if __name__ == "__main__":
    obj = DataLoader()
    obj.clean_data()