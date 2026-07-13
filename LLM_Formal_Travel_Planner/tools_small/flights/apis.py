import pandas as pd
from pandas import DataFrame
from typing import Optional
from utils.func import extract_before_parenthesis
from z3 import *
import numpy as np
import copy

class Flights:

    def __init__(self, path="database_small/flights/all.csv"):
        self.path = path
        self.data = None
        # Origin,Destination,Sun,Mon,Tu,Wed,Thu,Fri,Sat,Airline,Time,Duration,stop,new day
        self.data = pd.read_csv(self.path).dropna()[['Origin', 'Destination', 'Sun','Mon','Tu','Wed','Thu','Fri','Sat', 'Airline', 'Time','Duration','non-stop','new day']]
        print("Flights API loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path).dropna().rename(columns={'Unnamed: 0': 'Flight Number'})

    def run_info(self,
            origin: str,
            destination: str,
            ) -> DataFrame:
        """Search for flights by origin, destination, and departure date."""
        results = self.data[self.data["Origin"] == origin]
        results = results[results["Destination"] == destination]
        # import pdb; pdb.set_trace()
        if len(results) == 0:
            return "There is no flight from {} to {}.".format(origin, destination)
        return results[['Origin', 'Destination', 'Airline', 'Time', 'Duration', 'non-stop']]
    
    def get_airline_flights(self,
            airline: str,
            ) -> DataFrame:
        results = self.data[self.data["Airline"] == airline]
        # the results should show the index
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return f"There is no {airline} in all cities."
        return results[['Origin', 'Destination', 'Airline', 'Time', 'Duration', 'non-stop']]


    def run(self,
            origin: str,
            destination: str,
            departure_date: str,
            ) -> DataFrame:
        """Search for flights by origin, destination, and departure date."""
        results = self.data[self.data["Origin"] == origin]
        results = results[results["Destination"] == destination]
        weekdays = ['Sun','Mon','Tu','Wed','Thu','Fri','Sat']
        dates = ["2023-12-24", "2023-12-25", "2023-12-26", "2023-12-27", "2023-12-28", "2023-12-29", "2023-12-30"]
        # results = results[weekdays[dates.index(departure_date)]]
        if len(results) == 0:
            return "There is no flight from {} to {} on {}.".format(origin, destination, departure_date)
        return results

    def run_for_all_cities_and_dates(self,
            origin: str,
            all_cities: list,
            cities_list: list,
            departure_dates: list,
            ) -> DataFrame:
        """Search for flights by origin, destination, and departure date."""
        def convert_time(times):
            depart = []
            arrive = []
            for time in times:
                dep = time.split('-')[0]
                # import pdb; pdb.set_trace()
                hour = dep.split(':')[0]
                minute = dep.split(':')[1]
                time_float = int(hour) + float(minute)/60
                depart.append(time_float)
                arr = time.split('-')[1]
                hour = arr.split(':')[0]
                minute = arr.split(':')[1]
                time_float = int(hour) + float(minute)/60
                arrive.append(time_float)
            return depart, arrive
        weekdays = ['Sun','Mon','Tu','Wed','Thu','Fri','Sat']
        dates = ["2023-12-24", "2023-12-25", "2023-12-26", "2023-12-27", "2023-12-28", "2023-12-29", "2023-12-30"]
        cities = copy.deepcopy(cities_list)
        cities.insert(0, origin)
        all_cities = copy.deepcopy(all_cities)
        all_cities.insert(0, origin)
        results = Array('flights', IntSort(), IntSort(), IntSort(), IntSort(), ArraySort(IntSort(), RealSort())) # ori, dest, date, [Price, DepTime, ArrTime], info
        results_rules = Array('flights rules', IntSort(), IntSort(), ArraySort(IntSort(), BoolSort()))
        results_airlines = Array('flights airlines', IntSort(), IntSort(), ArraySort(IntSort(), IntSort(), BoolSort()))
        airlines_list = np.unique(self.data['Airline'].to_numpy())
        for i, ori in enumerate(cities):
            if i!= len(cities)-1:
                destination = cities[i+1]
            else:
                destination = cities[0]
            for d, departure_date in enumerate(departure_dates):
                # print(d)
                date = dates.index(departure_date)
                # print(origin, destination)
                result = self.data[self.data["Origin"] == ori]
                # print(result)
                result = result[result["Destination"] == destination]
                if len(result) != 0:
                    price = Array('Price', IntSort(), RealSort())
                    depTime = Array('DepTime', IntSort(), RealSort())
                    arrTime = Array('ArrTime', IntSort(), RealSort())
                    length = Array('Length', IntSort(), RealSort())
                    DepTime, ArrTime = convert_time(np.array(result)[:,10])
                    length = Store(length, 0, len(np.array(result)[:,1]))
                    # print(len(np.array(result)[:,1]))
                    rules = Array('Rules', IntSort(), BoolSort())
                    airlines = Array('Airlines', IntSort(), IntSort(), BoolSort())
                    for index in range(np.array(result).shape[0]):
                        # print(np.array(result)[:,date+3][index])
                        price = Store(price, index, np.array(result)[:,date+2][index])
                        depTime = Store(depTime, index, DepTime[index])
                        arrTime = Store(arrTime, index, ArrTime[index])
                        rules = Store(rules, index, np.array(result)[:,12][index] == 'yes')
                        for j in range(len(airlines_list)):
                            airlines = Store(airlines, index, j, airlines_list[j] == np.array(result)[:,9][index])
                    results = Store(results, all_cities.index(ori), all_cities.index(destination), d, 0, price)
                    results = Store(results, all_cities.index(ori), all_cities.index(destination), d, 1, depTime)
                    results = Store(results, all_cities.index(ori), all_cities.index(destination), d, 2, arrTime)
                    results = Store(results, all_cities.index(ori), all_cities.index(destination), d, 3, length)
                    results_rules = Store(results_rules, all_cities.index(ori), all_cities.index(destination), rules)
                    results_airlines = Store(results_airlines, all_cities.index(ori), all_cities.index(destination), airlines)
                else:
                    # import pdb; pdb.set_trace()
                    length = Array('Length', IntSort(), RealSort())
                    length = Store(length, 0, -1)
                    results = Store(results, all_cities.index(ori), all_cities.index(destination), d, 3, length)
        # import pdb; pdb.set_trace()
        return results, results_rules, results_airlines
    

    def get_info(self, info, i, j, d, key):
        if type(i) == str and type(j) == str:
                i = 0
                j = 1
        elif type(i) == str:
            i = 0
            j += 1
        elif type(j) == str:
            j = 0
            i += 1
        else:
            i += 1
            j += 1
        if key == 'Flight rules' or key == 'Airlines':
            info_key = Select(info, i, j)
            return info_key, None
        else:
            element = ['Price', 'DepTime', 'ArrTime', 'Length']
            info_key = Select(info, i, j, d, element.index(key))
            info_length = Select(info, i, j, d, 3)
            length = Select(info_length, 0)
            return info_key, length

    def get_info_for_index(self, price_list, index):
        return Select(price_list, index)

    def check_exists(self, category, category_list, attraction_index):
        if category == 'non-stop':
            exists = Select(category_list, attraction_index)
            return If(attraction_index != -1, exists, False)
        else:
            categories = list(np.unique(self.data['Airline'].to_numpy()))
            # import pdb; pdb.set_trace()
            exists = Select(category_list, attraction_index, categories.index(category))
            return If(attraction_index != -1, exists, False)
    
    # def run_for_annotation(self,
    #         origin: str,
    #         destination: str,
    #         departure_date: str,
    #         ) -> DataFrame:
    #     """Search for flights by origin, destination, and departure date."""
    #     results = self.data[self.data["OriginCityName"] == extract_before_parenthesis(origin)]
    #     results = results[results["DestCityName"] == extract_before_parenthesis(destination)]
    #     results = results[results["FlightDate"] == departure_date]
    #     # if order == "ascPrice":
    #     #     results = results.sort_values(by=["Price"], ascending=True)
    #     # elif order == "descPrice":
    #     #     results = results.sort_values(by=["Price"], ascending=False)
    #     # elif order == "ascDepTime":
    #     #     results = results.sort_values(by=["DepTime"], ascending=True)
    #     # elif order == "descDepTime":
    #     #     results = results.sort_values(by=["DepTime"], ascending=False)
    #     # elif order == "ascArrTime":
    #     #     results = results.sort_values(by=["ArrTime"], ascending=True)
    #     # elif order == "descArrTime":
    #     #     results = results.sort_values(by=["ArrTime"], ascending=False)
    #     return results.to_string(index=False)

    # def get_city_set(self):
    #     city_set = set()
    #     for unit in self.data['data']:
    #         city_set.add(unit[5])
    #         city_set.add(unit[6])
