from pybehavior.tools import WeatherProcessor
import pandas as pd
from pyncei import NCEIBot, NCEIResponse
import logging

logging.basicConfig(filename='log.txt', encoding='utf-8', level=logging.DEBUG)

weather = WeatherProcessor("pOxgBkZdoFwzokPdlFajJcmzMySAOUkP")
weather.refresh_weather_info()
print(weather.weather_by_zipcode_db)
