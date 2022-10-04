import os
import time
import datetime
import logging
from turtle import update
import pytz
from datetime import date
from typing import List
import pandas as pd
from pandas import read_csv, to_datetime
from pandas.core.frame import DataFrame

from pyncei import NCEIBot, NCEIResponse



class Loader:
    def pick_columns(
        data:DataFrame,
        user:str,
        start_datetime:str,
        end_datetime:str,
        values:List[str]
    ) -> DataFrame:
        column_list = [user, start_datetime, end_datetime] + values
        return_data = data[column_list].rename(columns={user: 'user', start_datetime: 'start', end_datetime: 'end'})
        return_data['start'] = pd.to_datetime(return_data.start)
        return_data['end'] = pd.to_datetime(return_data.end)
        return_data = return_data.set_index(['user', 'start', 'end'])
        return return_data

class Preprocessor:
    def merge_rows(data: DataFrame):
        data = data.reset_index()
        data['shifted_end'] = data.groupby(['user'])[['end']].shift(fill_value=datetime.datetime(datetime.MAXYEAR, 12, 31, 23, 59))
        data['group'] = ((1 - (data['start'] == data['shifted_end'])).cumsum())
        data = data.groupby(['user', 'group']).agg({'start': 'first', 'end': 'last', 'steps': 'mean'}).reset_index()
        data['duration'] = data['end'] - data['start']
        data = data.sort_values(['user', 'start', 'end'])
        data = data.set_index(['user', 'start', 'end']).drop('group', axis=1)

        return data
    
    def get_resampled_data(data:pd.DataFrame, group:str=None, start='start', end='end', span_unit='hour', unit='hour'):
        if group:
            data = data[[group, start, end]].drop_duplicates()
        else:
            data = data[[start, end]].drop_duplicates()

        if span_unit == 'hour':
            data['starthour'] = pd.to_datetime(data[start].dt.date) + pd.to_timedelta(data[start].dt.hour, unit='hour')
            data['endhour'] = pd.to_datetime(data[end].dt.date) + pd.to_timedelta(data[end].dt.hour + 1, unit='hour')
        elif span_unit == 'day':
            data['starthour'] = pd.to_datetime(data[start].dt.date)
            data['endhour'] = pd.to_datetime(data[end].dt.date) + datetime.timedelta(days=1)

        if group:
            data = data[[group, 'starthour', 'endhour']].drop_duplicates()
        else:
            data = data[['starthour', 'endhour']].drop_duplicates()
        
        def t(row):
            if group:
                group_info = getattr(row, group)
            
            start_datetime = row.starthour
            end_datetime = row.endhour

            timedelta = (end_datetime - start_datetime).total_seconds()

            if unit == 'hour':
                timedelta = timedelta / 3600
                timefreq = 'H'
            elif unit == 'day':
                timedelta = timedelta / 86400
                timefreq = 'D'
            
            date_range = pd.date_range(start_datetime, periods=timedelta, freq=timefreq)
            if unit == 'hour':
                time_unit = datetime.timedelta(hours=1)
            elif unit == 'day':
                time_unit = datetime.timedelta(days=1)
            else:
                raise ValueError('Invalid unit: %s' % unit)

            data_dict = {
                    start: date_range,
                    end: date_range + time_unit
                }
            if group:
                data_dict[group] = group_info
            return pd.DataFrame(data_dict)
        
        data = pd.concat(data.apply(t, axis=1).tolist()).reset_index(drop=True)
        if group:
            column_order = [group, start, end]
        else:
            column_order = [start, end]
        data = data[column_order].drop_duplicates().sort_values(column_order).reset_index(drop=True)

        return data

    def get_hourly_activity_data(data):
        data = data.reset_index()
        hours_with_activity = Preprocessor.get_resampled_data(data=data, group='user', start='start', end='end', span_unit='hour', unit='hour')
        hours_with_activity['activity'] = 2

        hours_with_data = Preprocessor.get_resampled_data(data=data, group='user', start='start', end='end', span_unit='day', unit='hour')
        hours_with_12 = pd.merge(hours_with_activity, hours_with_data, on=['user', 'start', 'end'], how="outer").fillna(1)
        
        full_day_range = hours_with_data.groupby('user').agg({'start': min, 'end': max}).reset_index()
        hours_full_day_range = Preprocessor.get_resampled_data(full_day_range, group='user', start='start', end='end', span_unit='day', unit='hour')
        
        hours_with_012 = pd.merge(hours_with_12, hours_full_day_range, on=['user', 'start', 'end'], how='outer').fillna(0)
        hours_with_012['activity'] = hours_with_012['activity'].astype(int)
        hours_with_012 = hours_with_012.sort_values(['user', 'start', 'end']).reset_index(drop=True)
        
        return hours_with_012

    def fill_missing_dates(data, date, group=None):
        date_min = '{}_min'.format(date)
        date_max = '{}_max'.format(date)

        data[date] = pd.to_datetime(data[date])
        if group:
            date_span = data.groupby(group).agg({date: ['min', 'max']})
        else:
            date_span = data.groupby(group).agg({date: ['min', 'max']})
        
        date_span = date_span.reset_index()
        
        if group:
            date_span.columns = [group, date_min, date_max]
        else:
            date_span.columns = [date_min, date_max]

        every_day = Preprocessor.get_resampled_data(date_span, group=group, start=date_min, end=date_max, span_unit='day', unit='day')
        
        if group:
            every_day = every_day[[group, date_min]].rename(columns={date_min: date})
            return pd.merge(data, every_day, on=[group, date], how="outer", sort=True)
        else:
            every_day = every_day[[date_min]].rename(columns={date_min: date})
            return pd.merge(data, every_day, on=[date], how="outer", sort=True)
        
class WeatherProcessor:
    data_path = 'data/weather'
    location_db_columns = ['zipcode', 'station_id', 'station_lat', 'station_lon']
    weather_by_station_db_columns = ['station_id', 'datatype', 'date', 'value']

    def __init__(self, NCEI_token):
        if not os.path.exists(WeatherProcessor.data_path):
            os.makedirs(WeatherProcessor.data_path)
        
        self.__load_db()
        
        # NCEI Robot
        self.ncei = NCEIBot(NCEI_token, cache_name="ncei", wait=1)

        # Timezone
        self.tz = pytz.timezone('America/Los_Angeles')

        # Today
        self.when_created = self.tz.localize(datetime.datetime.now())
        self.today = self.when_created.date()

    def __get_db_path(self, db_name):
        db_path = os.path.join(WeatherProcessor.data_path, '{}_db.pickle'.format(db_name))
        return db_path

    def __safe_convert(self, response, columns:List[str]) -> pd.DataFrame:
        if response.count() == 1 and response.first() == {}:
            return pd.DataFrame(columns=columns)
        else:
            return response.to_dataframe()
        
        try:
            pass
        except:
            try:
                iterator = response.values()

                dict_list = []
                for item in iterator:
                    try:
                        print(item)
                        dict_list.append(item)
                    except:
                        pass
                print(dict_list)
                if len(dict_list) > 1 or (len(dict_list) == 1 and dict_list[0] != {}):
                    return pd.DataFrame(dict_list)
                else:
                    return pd.DataFrame(columns=columns)
            except:
                return pd.DataFrame(columns=columns)

    def __load_db(self):
        # Location DB
        location_db_path = self.__get_db_path('location')
        if os.path.exists(location_db_path):
            self.location_db = pd.read_pickle(location_db_path)
        else:
            self.location_db = pd.DataFrame(columns=WeatherProcessor.location_db_columns)
        
        # Weather by Station DB
        weather_by_station_db_path = self.__get_db_path('weather_by_station')
        if os.path.exists(weather_by_station_db_path):
            self.weather_by_station_db = pd.read_pickle(weather_by_station_db_path)
        else:
            self.weather_by_station_db = pd.DataFrame(columns=WeatherProcessor.weather_by_station_db_columns)
        
        # Weather Merge DB
        weather_merged_db_path = self.__get_db_path('weather_merged')
        if os.path.exists(weather_merged_db_path):
            self.weather_merged_db = pd.read_pickle(weather_merged_db_path)
        else:
            self.weather_merged_db = pd.merge(self.location_db, self.weather_by_station_db, on='station_id')

        # Weather By ZIP Code DB
        weather_by_zipcode_db_path = self.__get_db_path('weather_by_zipcode')
        if os.path.exists(weather_by_zipcode_db_path):
            self.weather_by_zipcode_db = pd.read_pickle(weather_by_zipcode_db_path)
        else:
            self.weather_by_zipcode_db = pd.pivot_table(self.weather_merged_db.loc[self.weather_merged_db['datatype'].isin(['TMAX', 'TMIN', 'PRCP', 'AWND'])], values='value', index=['zipcode', 'date'], columns='datatype', aggfunc='mean').reset_index().sort_values(['zipcode', 'date']).drop_duplicates()

    def __save_db(self):
        location_db_path = self.__get_db_path('location')
        self.location_db.to_pickle(location_db_path)

        weather_by_station_db_path = self.__get_db_path('weather_by_station')
        self.weather_by_station_db.to_pickle(weather_by_station_db_path)

        weather_merged_db_path = self.__get_db_path('weather_merged')
        self.weather_merged_db.to_pickle(weather_merged_db_path)

        weather_by_zipcode_db_path = self.__get_db_path('weather_by_zipcode')
        self.weather_by_zipcode_db.to_pickle(weather_by_zipcode_db_path)

    def add_zipcode(self, zipcode, lat, lon) -> 'WeatherProcessor':
        # check if zipcode is already in the database
        matched = self.location_db.query('zipcode == @zipcode')
        if matched.shape[0] == 0:
            # search only if zipcode is not in the database
            print("Search for the station in ZIPCODE {}".format(zipcode))
            stations_columns = ['id', 'latitude', 'longitude']
            gap = 0.01  # initial gap for lat/lon
            station_count = 0
            while station_count < 20:    # search until we get 5 stations
                min_lat, min_lon, max_lat, max_lon = lat - gap, lon - gap, lat + gap, lon + gap
                extent_str = "{},{},{},{}".format(min_lat, min_lon, max_lat, max_lon)
                print('  Searching in ({})'.format(extent_str))
                response = self.ncei.get_stations(extent=extent_str, startdate="2022-01-01")
                stations = self.__safe_convert(response, stations_columns)
                stations = stations[stations_columns].rename(columns={'id': 'station_id', 'latitude': 'station_lat', 'longitude': 'station_lon'})
                station_count = stations.shape[0]
                print('    -> {} station(s) found'.format(station_count))
                gap = gap * 1.5

            stations['zipcode'] = zipcode
            stations = stations[['zipcode', 'station_id', 'station_lat', 'station_lon']]
            
            self.location_db = pd.concat([self.location_db, stations], axis=0).reset_index(drop=True)
            self.__save_db()

        return self
    
    def refresh_weather_info(self):
        # per-station check
        station_list_in_location_db = self.location_db[['station_id']].drop_duplicates()
        station_list_in_weather_db = self.weather_by_station_db.groupby('station_id').agg({'date': 'max'}).reset_index()
        station_list_in_weather_db.columns = ['station_id', 'last_date']
        station_list = pd.merge(station_list_in_location_db, station_list_in_weather_db, on='station_id', how="outer")
        
        ## stations never updated
        never_updated = station_list.query('last_date.isnull()')

        response = self.ncei.get_data(datasetid='GHCND', stationid=never_updated['station_id'].to_list(), startdate="2022-01-01", enddate=self.today)
        response_df = self.__safe_convert(response, ['station', 'date', 'datatype', 'attribute', 'value', 'url', 'retrieved'])
        response_df = response_df[['station', 'datatype', 'date', 'value']].rename(columns={'station': 'station_id'})

        self.weather_by_station_db = pd.concat([self.weather_by_station_db, response_df], axis=0).reset_index(drop=True)
        
        ## the last update was too old
        target_max_date = self.today - datetime.timedelta(days=3)
        been_a_while = station_list.query('last_date < @target_max_date')

        if_updated = False
        for index, row in been_a_while.iterrows():
            response = self.ncei.get_data(datasetid='GHCND', stationid=row.station_id, startdate=(row.last_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"), enddate=self.today)
            response_df = self.__safe_convert(response, [])
            if response_df.shape[0] > 0:
                if_updated = True
                print(response_df)
                response_df = response_df[['station', 'datatype', 'date', 'value']].rename(columns={'station': 'station_id'})
                self.weather_by_station_db = pd.concat([self.weather_by_station_db, response_df], axis=0)
        
        self.weather_merged_db = pd.merge(self.location_db, self.weather_by_station_db, on='station_id')
        self.weather_by_zipcode_db = pd.pivot_table(self.weather_merged_db.loc[self.weather_merged_db['datatype'].isin(['TMAX', 'TMIN', 'PRCP', 'AWND'])], values='value', index=['zipcode', 'date'], columns='datatype', aggfunc='mean').reset_index().sort_values(['zipcode', 'date']).drop_duplicates()

        self.__save_db()

        return self