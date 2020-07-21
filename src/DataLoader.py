'''
Created on 6/21/20

@author: dulanj
'''
import os

import pandas as pd
import logging
from geopy.geocoders import Nominatim
from functools import partial


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
    location_file = 'locations.csv'

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.train_filename = DataLoader.train_filename
        self.test_filename = DataLoader.test_filename

        self.train_df = pd.read_csv(self.train_filename)
        self.test_df = pd.read_csv(self.test_filename)

        geolocator = Nominatim(user_agent="MSCinCS")
        self.reverse = partial(geolocator.reverse, language="en")


        self.location_dict = dict()
        self.load_to_dict()
        self.filepointer = open(DataLoader.location_file, 'a')
        self.get_loc_counter = 0

    def get_key(self, lat, lon):
        return '{}-{}'.format(lat,lon)

    def load_to_dict(self):
        if os.path.exists(DataLoader.location_file):
            print("Loading to dict")
            fp = open(DataLoader.location_file, 'r')
            for line in fp.readlines():
                print(line)
                lat, lon, loc = line.split(':')[:3]
                self.location_dict[self.get_key(lat, lon)] = loc

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

    def write_loc_to_csv(self, lat, lon, location_info):
        self.filepointer.write('{}:{}:{}\n'.format(lat, lon, location_info))
        print("{}:{}:{} - Written".format(lat, lon, location_info))

    def get_location(self, lat, lon):
        self.get_loc_counter += 1
        print("Getting location {}".format(self.get_loc_counter))
        _key = self.get_key(lat, lon)
        if _key in self.location_dict:
            return self.location_dict[_key]
        else:
            location_info = self.reverse("{}, {}".format(lat, lon))
            self.write_loc_to_csv(lat, lon, location_info)
            return location_info

    def geo_location(self):

        def get_loc(row):
            location = self.get_location(row[Columns.pick_lat], row[Columns.pick_lon])
            # print(str(location).split(',')[-5])
            return location

        self.train_df['pick_loc'] = self.train_df.apply(lambda row: get_loc(row), axis=1)
        self.test_df['pick_loc'] = self.test_df.apply(lambda row: get_loc(row), axis=1)




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
    obj.geo_location()
    # print(obj.train_df.iloc[41])