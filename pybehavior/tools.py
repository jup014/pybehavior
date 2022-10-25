import datetime
import os

import BayesLDM.BayesLDM as BayesLDM
import pandas as pd
import numpy as np
import pytz
from numpyro.infer import MCMC
from pandas.core.frame import DataFrame
from pyncei import NCEIBot


class Loader:
    @staticmethod
    def pick_columns(
        data: DataFrame,
        user: str,
        start_datetime: str,
        end_datetime: str,
        values,
    ) -> DataFrame:
        column_list = [user, start_datetime, end_datetime] + values
        return_data = data[column_list].rename(
            columns={user: "user", start_datetime: "start", end_datetime: "end"}
        )
        return_data["start"] = pd.to_datetime(return_data.start)
        return_data["end"] = pd.to_datetime(return_data.end)
        return_data = return_data.set_index(["user", "start", "end"])
        return return_data


class Preprocessor:
    @staticmethod
    def merge_rows(data: DataFrame):
        data = data.reset_index()
        data["shifted_end"] = data.groupby(["user"])[["end"]].shift(
            fill_value=datetime.datetime(datetime.MAXYEAR, 12, 31, 23, 59)
        )
        data["group"] = (1 - (data["start"] == data["shifted_end"])).cumsum()
        data = (
            data.groupby(["user", "group"])
            .agg({"start": "first", "end": "last", "steps": "mean"})
            .reset_index()
        )
        data["duration"] = data["end"] - data["start"]
        data = data.sort_values(["user", "start", "end"])
        data = data.set_index(["user", "start", "end"]).drop("group", axis=1)

        return data

    @staticmethod
    def get_resampled_data(
        data: pd.DataFrame,
        group = None,
        start="start",
        end="end",
        span_unit="hour",
        unit="hour",
    ):
        if group:
            data = data[[group, start, end]].drop_duplicates()
        else:
            data = data[[start, end]].drop_duplicates()

        if span_unit == "hour":
            data["starthour"] = pd.to_datetime(data[start].dt.date) + pd.to_timedelta(
                data[start].dt.hour, unit="hour"
            )
            data["endhour"] = pd.to_datetime(data[end].dt.date) + pd.to_timedelta(
                data[end].dt.hour + 1, unit="hour"
            )
        elif span_unit == "day":
            data["starthour"] = pd.to_datetime(data[start].dt.date)
            data["endhour"] = pd.to_datetime(data[end].dt.date).add(
                datetime.timedelta(days=1)
            )

        if group:
            data = data[[group, "starthour", "endhour"]].drop_duplicates()
        else:
            data = data[["starthour", "endhour"]].drop_duplicates()

        def t(row):
            if group:
                group_info = getattr(row, group)
            else:
                group_info = None

            start_datetime = row.starthour
            end_datetime = row.endhour

            timedelta = (end_datetime - start_datetime).total_seconds()

            if unit == "hour":
                timedelta = timedelta / 3600
                timefreq = "H"
            elif unit == "day":
                timedelta = timedelta / 86400
                timefreq = "D"
            else:
                raise ValueError("Invalid unit")

            date_range = pd.date_range(start_datetime, periods=timedelta, freq=timefreq)
            if unit == "hour":
                time_unit = datetime.timedelta(hours=1)
            elif unit == "day":
                time_unit = datetime.timedelta(days=1)
            else:
                raise ValueError("Invalid unit: %s" % unit)

            data_dict = {start: date_range, end: date_range.to_pydatetime() + time_unit}
            if group:
                data_dict[group] = group_info
            return pd.DataFrame(data_dict)

        data = pd.concat(data.apply(t, axis=1).tolist()).reset_index(drop=True)
        if group:
            column_order = [group, start, end]
        else:
            column_order = [start, end]
        data = (
            data[column_order]
            .drop_duplicates()
            .sort_values(column_order)
            .reset_index(drop=True)
        )

        return data

    @staticmethod
    def get_hourly_activity_data(data):
        data = data.reset_index()
        hours_with_activity = Preprocessor.get_resampled_data(
            data=data,
            group="user",
            start="start",
            end="end",
            span_unit="hour",
            unit="hour",
        )
        hours_with_activity["activity"] = 2

        hours_with_data = Preprocessor.get_resampled_data(
            data=data,
            group="user",
            start="start",
            end="end",
            span_unit="day",
            unit="hour",
        )
        hours_with_12 = pd.merge(
            hours_with_activity,
            hours_with_data,
            on=["user", "start", "end"],
            how="outer",
        ).fillna(1)

        full_day_range = (
            hours_with_data.groupby("user")
            .agg({"start": min, "end": max})
            .reset_index()
        )
        hours_full_day_range = Preprocessor.get_resampled_data(
            full_day_range,
            group="user",
            start="start",
            end="end",
            span_unit="day",
            unit="hour",
        )

        hours_with_012 = pd.merge(
            hours_with_12,
            hours_full_day_range,
            on=["user", "start", "end"],
            how="outer",
        ).fillna(0)
        hours_with_012["activity"] = hours_with_012["activity"].astype(int)
        hours_with_012 = hours_with_012.sort_values(
            ["user", "start", "end"]
        ).reset_index(drop=True)

        return hours_with_012

    @staticmethod
    def fill_missing_dates(data: pd.DataFrame, date, group=None):
        date_min = "{}_min".format(date)
        date_max = "{}_max".format(date)

        data[date] = pd.to_datetime(data[date])
        if group:
            date_span = data.groupby(group).agg({date: ["min", "max"]})
        else:
            date_span = data.groupby(group).agg({date: ["min", "max"]})

        date_span = date_span.reset_index()

        if group:
            date_span.columns = [group, date_min, date_max]
        else:
            date_span.columns = [date_min, date_max]

        every_day = Preprocessor.get_resampled_data(
            date_span,
            group=group,
            start=date_min,
            end=date_max,
            span_unit="day",
            unit="day",
        )

        if group:
            every_day = every_day[[group, date_min]].rename(columns={date_min: date})
            return pd.merge(data, every_day, on=[group, date], how="outer", sort=True)
        else:
            every_day = every_day[[date_min]].rename(columns={date_min: date})
            return pd.merge(data, every_day, on=[date], how="outer", sort=True)


class WeatherProcessor:
    data_path = "data/weather"
    location_db_columns = ["zipcode", "station_id", "station_lat", "station_lon"]
    weather_by_station_db_columns = ["station_id", "datatype", "date", "value"]

    def __init__(self, NCEI_token):
        if not os.path.exists(WeatherProcessor.data_path):
            os.makedirs(WeatherProcessor.data_path)

        self.__load_db()

        # NCEI Robot
        self.ncei = NCEIBot(NCEI_token, cache_name="ncei", wait=1)

        # Timezone
        self.tz = pytz.timezone("America/Los_Angeles")

        # Today
        self.when_created = self.tz.localize(datetime.datetime.now())
        self.today = self.when_created.date()

    def __get_db_path(self, db_name):
        db_path = os.path.join(
            WeatherProcessor.data_path, "{}_db.pickle".format(db_name)
        )
        return db_path

    def __safe_convert(self, response, columns) -> pd.DataFrame:
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
        location_db_path = self.__get_db_path("location")
        if os.path.exists(location_db_path):
            self.location_db = pd.read_pickle(location_db_path)
        else:
            self.location_db = pd.DataFrame(
                columns=WeatherProcessor.location_db_columns
            )

        # Weather by Station DB
        weather_by_station_db_path = self.__get_db_path("weather_by_station")
        if os.path.exists(weather_by_station_db_path):
            self.weather_by_station_db = pd.read_pickle(weather_by_station_db_path)
        else:
            self.weather_by_station_db = pd.DataFrame(
                columns=WeatherProcessor.weather_by_station_db_columns
            )

        # Weather Merge DB
        weather_merged_db_path = self.__get_db_path("weather_merged")
        if os.path.exists(weather_merged_db_path):
            self.weather_merged_db = pd.read_pickle(weather_merged_db_path)
        else:
            self.weather_merged_db = pd.merge(
                self.location_db, self.weather_by_station_db, on="station_id"
            )

        # Weather By ZIP Code DB
        weather_by_zipcode_db_path = self.__get_db_path("weather_by_zipcode")
        if os.path.exists(weather_by_zipcode_db_path):
            self.weather_by_zipcode_db = pd.read_pickle(weather_by_zipcode_db_path)
        else:
            self.weather_by_zipcode_db = (
                pd.pivot_table(
                    self.weather_merged_db.loc[
                        self.weather_merged_db["datatype"].isin(
                            ["TMAX", "TMIN", "PRCP", "AWND"]
                        )
                    ],
                    values="value",
                    index=["zipcode", "date"],
                    columns="datatype",
                    aggfunc="mean",
                )
                .reset_index()
                .sort_values(["zipcode", "date"])
                .drop_duplicates()
            )

    def __save_db(self):
        location_db_path = self.__get_db_path("location")
        self.location_db.to_pickle(location_db_path)

        weather_by_station_db_path = self.__get_db_path("weather_by_station")
        self.weather_by_station_db.to_pickle(weather_by_station_db_path)

        weather_merged_db_path = self.__get_db_path("weather_merged")
        self.weather_merged_db.to_pickle(weather_merged_db_path)

        weather_by_zipcode_db_path = self.__get_db_path("weather_by_zipcode")
        self.weather_by_zipcode_db.to_pickle(weather_by_zipcode_db_path)

    def add_zipcode(self, zipcode, lat, lon) -> "WeatherProcessor":
        # check if zipcode is already in the database
        matched = self.location_db.query("zipcode == @zipcode")
        if matched.shape[0] == 0:
            # search only if zipcode is not in the database
            print("Search for the station in ZIPCODE {}".format(zipcode))
            stations_columns = ["id", "latitude", "longitude"]
            gap = 0.01  # initial gap for lat/lon
            station_count = 0
            while station_count < 20:  # search until we get 5 stations
                min_lat, min_lon, max_lat, max_lon = (
                    lat - gap,
                    lon - gap,
                    lat + gap,
                    lon + gap,
                )
                extent_str = "{},{},{},{}".format(min_lat, min_lon, max_lat, max_lon)
                print("  Searching in ({})".format(extent_str))
                response = self.ncei.get_stations(
                    extent=extent_str, startdate="2022-01-01"
                )
                stations = self.__safe_convert(response, stations_columns)
                stations = stations[stations_columns].rename(
                    columns={
                        "id": "station_id",
                        "latitude": "station_lat",
                        "longitude": "station_lon",
                    }
                )
                station_count = stations.shape[0]
                print("    -> {} station(s) found".format(station_count))
                gap = gap * 1.5

            stations["zipcode"] = zipcode
            stations = stations[["zipcode", "station_id", "station_lat", "station_lon"]]

            self.location_db = pd.concat(
                [self.location_db, stations], axis=0
            ).reset_index(drop=True)
            self.__save_db()

        return self

    def refresh_weather_info(self):
        # per-station check
        station_list_in_location_db = self.location_db[["station_id"]].drop_duplicates()
        station_list_in_weather_db = (
            self.weather_by_station_db.groupby("station_id")
            .agg({"date": "max"})
            .reset_index()
        )
        station_list_in_weather_db.columns = ["station_id", "last_date"]
        station_list = pd.merge(
            station_list_in_location_db,
            station_list_in_weather_db,
            on="station_id",
            how="outer",
        )

        ## stations never updated
        never_updated = station_list.query("last_date.isnull()")

        response = self.ncei.get_data(
            datasetid="GHCND",
            stationid=never_updated["station_id"].to_list(),
            startdate="2022-01-01",
            enddate=self.today,
        )
        response_df = self.__safe_convert(
            response,
            ["station", "date", "datatype", "attribute", "value", "url", "retrieved"],
        )
        response_df = response_df[["station", "datatype", "date", "value"]].rename(
            columns={"station": "station_id"}
        )

        self.weather_by_station_db = pd.concat(
            [self.weather_by_station_db, response_df], axis=0
        ).reset_index(drop=True)

        ## the last update was too old
        target_max_date = self.today - datetime.timedelta(days=3)
        been_a_while = station_list.query("last_date < @target_max_date")

        if_updated = False
        for index, row in been_a_while.iterrows():
            response = self.ncei.get_data(
                datasetid="GHCND",
                stationid=row.station_id,
                startdate=(row.last_date + datetime.timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                ),
                enddate=self.today,
            )
            response_df = self.__safe_convert(response, [])
            if response_df.shape[0] > 0:
                if_updated = True
                print(response_df)
                response_df = response_df[
                    ["station", "datatype", "date", "value"]
                ].rename(columns={"station": "station_id"})
                self.weather_by_station_db = pd.concat(
                    [self.weather_by_station_db, response_df], axis=0
                )

        self.weather_merged_db = pd.merge(
            self.location_db, self.weather_by_station_db, on="station_id"
        )
        self.weather_by_zipcode_db = (
            pd.pivot_table(
                self.weather_merged_db.loc[
                    self.weather_merged_db["datatype"].isin(
                        ["TMAX", "TMIN", "PRCP", "AWND"]
                    )
                ],
                values="value",
                index=["zipcode", "date"],
                columns="datatype",
                aggfunc="mean",
            )
            .reset_index()
            .sort_values(["zipcode", "date"])
            .drop_duplicates()
        )

        self.__save_db()

        return self


class MCMC_Parser:
    def __init__(self, model: BayesLDM.compile):
        assert model is not None, "model must be not None"
        mcmc = model.fitted_mcmc
        assert mcmc is not None, "model.fitted_mcmc must be not None"
        samples = mcmc.get_samples()
        assert samples is not None, "samples must be not None"

        keys = samples.keys()

        self.coefficient = {}
        self.variable = {}

        for each_key in keys:
            bracket_index = each_key.find("[")
            if bracket_index == -1:
                # non-indexed key
                self.coefficient[each_key] = np.array(samples[each_key]).mean()
            else:
                key_name = each_key[:bracket_index]
                closing_bracket_index = each_key.find("]")
                index_str = each_key[bracket_index + 1 : closing_bracket_index]
                if key_name not in self.variable:
                    self.variable[key_name] = {}
                self.variable[key_name][int(index_str)] = np.array(
                    samples[each_key]
                ).mean()


class BayesModelBuilder:
    def __init__(self, name):
        self.name = name
        self.variables = []
        self.input_list = None
        self.index = None

    def __str__(self):
        return self.__get_program_name()

    def __get_program_name(self):
        return "ProgramName: {}".format(self.name)

    def __get_distribution_str(self, variable_dict):
        distribution = variable_dict["distribution"]
        if distribution == "normal":
            return "N({},{})".format(variable_dict["mean"], variable_dict["stdev"])
        elif distribution == "exponential":
            return "Exp({})".format(variable_dict["exponent"])
        else:
            raise Exception(
                "Unexpected distribution: {}".format(variable_dict["distribution"])
            )

    def __get_variable_declaration(self):
        variable_declaration_lines = []

        for each_variable in self.variables:
            variable_declaration_lines.append(
                "{} ~ {}".format(
                    each_variable["name"], self.__get_distribution_str(each_variable)
                )
            )

        return "\n".join(variable_declaration_lines)

    def __get_index_declaration(self):
        if self.index:
            return "Indices: {} {} {}".format(
                self.index["name"], self.index["min"], self.index["max"]
            )
        else:
            return ""

    def __get_input_declaration(self):
        if self.input_list:
            return "Inputs: {}".format(", ".join(self.input_list))
        else:
            return ""

    def get_full_model(self):
        full_model_lines = []
        full_model_lines.append(self.__get_program_name())
        full_model_lines.append("")
        full_model_lines.append(self.__get_index_declaration())
        full_model_lines.append("")
        full_model_lines.append(self.__get_input_declaration())
        full_model_lines.append("")
        full_model_lines.append(self.__get_variable_declaration())

        return "\n".join(full_model_lines)

    def add_variable(
        self,
        name: str,
        distribution,
        mean = 0,
        stdev = 1,
        exponent = 0.1,
    ):
        if distribution == "normal":
            self.variables.append(
                {
                    "name": name,
                    "distribution": distribution,
                    "mean": mean,
                    "stdev": stdev,
                }
            )
        elif distribution == "exponential":
            self.variables.append(
                {"name": name, "distribution": distribution, "exponent": exponent}
            )
        else:
            raise Exception("Unexpected distribution type: {}".format(distribution))

    def add_variable_autoregressive(
        self,
        name:str,
        index:str="t",
        distribution="normal",
        gain=0.9,
        bias=0,
        stdev=1,
        exponent=0.1,
    ):
        self.add_variable(
            name="{}[{}]".format(name, index),
            distribution=distribution,
            mean="{}*{}[{}-1]+{}".format(gain, name, index, bias),
            stdev=stdev,
            exponent=exponent,
        )

    def add_variable_regression_edge(
        self,
        edge_list,
        index: str='t'
    ):
        node_list = []
        src_dict = {}

        for each_edge in edge_list:
            src, dest = each_edge
            node_list.append(src)
            node_list.append(dest)

            if dest not in src_dict:
                src_dict[dest] = []
            
            src_dict[dest].append(src)

        node_list = list(set(node_list))

        for node in node_list:
            self.add_variable("g_{}".format(node), "normal", 0, 1)  # gain
            self.add_variable("b_{}".format(node), "normal", 0, 1)  # gain
            self.add_variable("s_{}".format(node), "exponential", 0, 1)  # gain
            self.add_variable("{}[0]".format(node), "normal", 0, 1)
        
        for dest in src_dict.keys():
            src_list = src_dict[dest]
            for src in src_list:
                if src != dest:
                    self.add_variable("tau_{}_{}".format(src, dest), "normal", 0, 1)
            
            term_list = []

            autoregressive = False
            for src in src_list:
                if src != dest:
                    term_list.append(
                        "tau_{}_{} * {}[{}]".format(src, dest, src, index)
                    )
                else:
                    autoregressive = True

            if autoregressive:
                if len(term_list) > 0:
                    term_str = 'g_{}*{}[{}-1] + ({}) + b_{}'.format(dest, dest, index, " + ".join(term_list), dest)
                else:
                    term_str = 'g_{}*{}[{}-1] + b_{}'.format(dest, dest, index, dest)
            else:
                if len(term_list) > 0:
                    term_str = '({}) + b_{}'.format(dest, dest, index, " + ".join(term_list), dest)
                else:
                    term_str = 'b_{}'.format(dest)
            self.add_variable(
                name="{}[{}]".format(dest, index), 
                distribution='normal', 
                mean=term_str,
                stdev="s_{}".format(dest)
            )
                
    def set_index(self, name: str, min: int, max: int):
        self.index = {"name": name, "min": min, "max": max}

    def set_input(self, input_name_list):
        self.input_list = input_name_list
