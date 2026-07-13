import pandas as pd
from pandas import DataFrame
from typing import Optional
from utils.func import extract_before_parenthesis
from z3 import *
import numpy as np

class Attractions:
    def __init__(self, path="database/attractions/attractions.csv"):
        self.path = path
        self.data = pd.read_csv(self.path).dropna()[['Name','Latitude','Longitude','Address','Phone','Website',"City"]]
        print("Attractions loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path)

    def run(self,
            city: str,
            ) -> DataFrame:
        """Search for Accommodations by city and date."""
        results = self.data[self.data["City"] == city]
        # the results should show the index
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return "There is no attraction in this city."
        return results  
    
    def run_for_all_cities(self, all_cities,
            cities: list,
            ) -> DataFrame:
        """Search for flights by origin, destination, and departure date."""
        results = Array('attractions', IntSort(), IntSort()) # ori, dest, date, [Price, DepTime, ArrTime], info
        for i, city in enumerate(cities):
            result = self.data[self.data["City"] == city]
            if len(result) != 0:
                # print('attraction', city, len(result), len(np.array(result)[:,1]))
                results = Store(results, all_cities.index(city), IntVal(len(np.array(result)[:,1])))
            else:
                results = Store(results, all_cities.index(city), -1)
        return results

    def get_info(self, info, i):
        length = Select(info, i)
        return length
    
    def get_info_for_index(self, price_list, index):
        return Select(price_list, index)

    def attraction_in_which_city(self, arrives, origin, cities, departure_dates, days):
        result = []
        origin = -1
        cities = [origin] + cities
        arrives_array = Array('arrives', IntSort(), RealSort())
        cities_array = Array('cities', IntSort(), IntSort())
        departure_dates_array = Array('departure_dates', IntSort(), IntSort())
        for index, arrive in enumerate(arrives):
            arrives_array = Store(arrives_array, index, arrive)
        for index, city in enumerate(cities):
            cities_array = Store(cities_array, index, city)
        for index, date in enumerate(departure_dates):
            departure_dates_array = Store(departure_dates_array, index, date)
        i = 0
        for day in range(days):
            arrtime = Select(arrives_array, i)
            result.append(If(day == Select(departure_dates_array, i), If(arrtime > 18, Select(cities_array, i), Select(cities_array, i+1)), Select(cities_array, i+1)))
            i += If(day == Select(departure_dates_array, i), 1, 0)
        print("Having attraction_in_which_city info for {} attractions".format(len(result)))
        return result
      
    def run_for_annotation(self,
            city: str,
            ) -> DataFrame:
        """Search for Accommodations by city and date."""
        results = self.data[self.data["City"] == extract_before_parenthesis(city)]
        # the results should show the index
        results = results.reset_index(drop=True)
        return results