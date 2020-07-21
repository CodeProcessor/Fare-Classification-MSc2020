'''
Created on 6/21/20

@author: dulanj
'''
import os

import pandas as pd
import logging
from geopy.geocoders import Nominatim
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool


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

        train_df = pd.read_csv(self.train_filename)
        print(train_df.shape)
        train_df.dropna(inplace=True)
        print(train_df.shape)

        test_df = pd.read_csv(self.test_filename)

        self.concat_df = pd.concat([train_df, test_df])
        
        print(self.concat_df.head())
        print(self.concat_df.shape)
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
                # print(line)
                lat, lon = line.split(':')[:2]
                loc = ' '.join(line.split(':')[2:])
                # print(loc)
                self.location_dict[self.get_key(lat, lon)] = loc

    def get_dataframes(self):
        """
        Load dataframe
        """

        train_df = self.concat_df.iloc[0:16968]
        test_df = self.concat_df.iloc[16968:]
        test_df = test_df.drop([Columns.label], axis=1)
        test_df = test_df.fillna(0)
        train_df = train_df.fillna(0)
        logging.info("Test DF loaded\n{}\n".format(test_df.head()))
        logging.info("Train DF loaded\n{}\n".format(train_df.head()))
        print(train_df.columns)
        print(test_df.columns)
        return train_df, test_df

    def clean_data(self):
        """:arg
        Cleaning the data
        Drop NULL values
        """
        pass
        # logging.info("Length of data: {}".format(len(self.train_df[Columns.trip_id].values)))
        # logging.info("Dropping null rows")
        # self.train_df.dropna(inplace=True)
        # logging.info("Length of data: {}".format(len(self.train_df[Columns.trip_id].values)))

    def write_loc_to_csv(self, lat, lon, location_info):
        self.filepointer.write('{}:{}:{}\n'.format(lat, lon, location_info))
        print("{}:{}:{} - Written".format(lat, lon, location_info))

    def get_location(self, lat, lon):
        self.get_loc_counter += 1
        print("Getting location {}".format(self.get_loc_counter))
        _key = self.get_key(lat, lon)
        if _key in self.location_dict:
            return lat, lon, self.location_dict[_key]
        else:
            location_info = self.reverse("{}, {}".format(lat, lon))
            self.write_loc_to_csv(lat, lon, location_info)
            return lat, lon, location_info

    def geo_location(self):
        def get_loc(row, index):
            lat, lon, location = self.get_location(row[Columns.pick_lat], row[Columns.pick_lon])

            try:
                loc = str(location).split(',')[-index]
                print(loc)
            except IndexError:
                print(location)
                loc = str(location).split(',')[-index+1]

            return loc

        train_pickup = self.concat_df.apply(lambda row: get_loc(row, index=4), axis=1)
        train_pickup2 = self.concat_df.apply(lambda row: get_loc(row, index=5), axis=1)
        self.concat_df = pd.concat([self.concat_df, pd.get_dummies(train_pickup, prefix='pick_loc4_')], axis=1)
        self.concat_df = pd.concat([self.concat_df, pd.get_dummies(train_pickup2, prefix='pick_loc5_')], axis=1)

    def straight_distance(self):
        """:arg
        Taking the distance between pickup and drop locations
        """
        def get_dist(row):
            dist = ((row[Columns.pick_lat] - row[Columns.drop_lat])**2 + (row[Columns.pick_lon] - row[Columns.drop_lon])**2)
            return dist

        self.concat_df['dist'] = self.concat_df.apply(lambda row: get_dist(row), axis=1)
        # self.test_df['dist'] = self.test_df.apply(lambda row: get_dist(row), axis=1)
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

        train_surge = self.concat_df.apply(lambda row: get_rejects_percentage(row), axis=1)
        # test_surge = self.concat_df.apply(lambda row: get_rejects_percentage(row), axis=1)
        self.concat_df = pd.concat([self.concat_df, pd.get_dummies(train_surge, prefix='surge')], axis=1)
        # self.test_df = pd.concat([self.test_df, pd.get_dummies(test_surge, prefix='surge')], axis=1)


if __name__ == "__main__":
    obj = DataLoader()
    obj.geo_location()
    obj.get_dataframes()
    # obj.get_locations_parallel()
    # print(obj.train_df.iloc[41])