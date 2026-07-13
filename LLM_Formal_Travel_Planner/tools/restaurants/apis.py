import pandas as pd
from pandas import DataFrame
from typing import Optional
from utils.func import extract_before_parenthesis
from z3 import *
import numpy as np

class Restaurants:
    def __init__(self, path="database/restaurants/clean_restaurant_2022.csv"):
        self.path = path
        self.data = pd.read_csv(self.path).dropna()[['Name','Average Cost','Cuisines','Aggregate Rating','City']]
        print("Restaurants loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path).dropna()

    def run(self,
            city: str,
            ) -> DataFrame:
        """Search for restaurant ."""
        results = self.data[self.data["City"] == city]
        if len(results) == 0:
            return "There is no restaurant in this city."
        non_repeat = []
        drop_index = []
        for index in range(np.array(results).shape[0]):
            name = np.array(results)[:,0][index]
            if name not in non_repeat:
                non_repeat.append(name)
            else:
                drop_index.append(results.index.to_numpy()[index])
        return results.drop(drop_index)#.to_string()
    
    def run_for_all_cities(self, all_cities: list,
            cities: list,
            ) -> DataFrame:
        """Search for flights by origin, destination, and departure date."""
        results = Array('restaurant', IntSort(), IntSort(), ArraySort(IntSort(), IntSort())) # ori, dest, date, [Price, DepTime, ArrTime], info
        results_cuisines = Array('restaurant cuisines', IntSort(), ArraySort(IntSort(), IntSort(), BoolSort())) # ori, dest, date, [Price, DepTime, ArrTime], info
        cuisines_list = ['Chinese', 'American', 'Italian', 'Mexican', 'Indian', 'Mediterranean', 'French']
        for i, city in enumerate(cities):
            result = self.data[self.data["City"] == city]
            if len(result) != 0:
                # import pdb; pdb.set_trace()
                price = Array('Price', IntSort(), IntSort())
                cuisines = Array('Cuisines', IntSort(), IntSort(), BoolSort())
                length = Array('Length', IntSort(), IntSort())
                non_repeat = []
                non_repeat_index = []
                for index in range(np.array(result).shape[0]):
                    name = np.array(result)[:,0][index]
                    if name not in non_repeat:
                        non_repeat.append(name)
                        non_repeat_index.append(index)
                for order, index in enumerate(non_repeat_index):
                    price = Store(price, order, np.array(result)[:,1][index])
                    types = np.array(result)[:,2][index]
                    for j in range(len(cuisines_list)):
                        cuisines = Store(cuisines, order, j, cuisines_list[j] in types)

                length = Store(length, 0, len(non_repeat_index))
                results = Store(results, all_cities.index(city), 0, price)
                results = Store(results, all_cities.index(city), 1, length)
                # print('length!!!', length)
                results_cuisines = Store(results_cuisines, all_cities.index(city), cuisines)
            else:
                length = Array('Length', IntSort(), IntSort())
                length = Store(length, 0, -1)
                results = Store(results, all_cities.index(city), 1, length)
        return results, results_cuisines
    
    def get_info(self, info, i, key):
        # ['Price', 'Length']
        if key == 'Cuisines':
            info_key = Select(info, i)
            return info_key, None
        else:
            element = ['Price', 'Length']
            info_key = Select(info, i, element.index(key))
            info_length = Select(info, i, 1)
            length = Select(info_length, 0)
            return info_key, length
    
    def get_info_for_index(self, price_list, index):
        return Select(price_list, index)

    def eat_in_which_city(self, arrives, origin, cities, departure_dates, days):
        result = []
        origin = -1
        cities = [origin] + cities + [origin] 
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
        # arrtime = Select(arrives_array, 0)
        for day in range(days):
            arrtime = Select(arrives_array, i) #If(day == Select(departure_dates_array, i), Select(arrives_array, i), arrtime)
            result.append(If(day == Select(departure_dates_array, i), If(arrtime > 10, Select(cities_array, i), Select(cities_array, i+1)), Select(cities_array, i+1)))
            result.append(If(day == Select(departure_dates_array, i), If(arrtime > 13, Select(cities_array, i), Select(cities_array, i+1)), Select(cities_array, i+1)))
            result.append(If(day == Select(departure_dates_array, i), If(arrtime > 20, Select(cities_array, i), Select(cities_array, i+1)), Select(cities_array, i+1)))
            i += If(day == Select(departure_dates_array, i), 1, 0)
        print("Having eat in which info for {} restaurants".format(len(result)))
        # import pdb; pdb.set_trace()
        return result
    
    def check_exists(self, cuisine, restaurant_cuisines_list, restaurant_index):
        cuisines_list = ['Chinese', 'American', 'Italian', 'Mexican', 'Indian', 'Mediterranean', 'French']
        exists = Select(restaurant_cuisines_list, restaurant_index, cuisines_list.index(cuisine))
        return If(restaurant_index != -1, exists, False)

    def run_for_annotation(self,
            city: str,
            ) -> DataFrame:
        """Search for restaurant ."""
        results = self.data[self.data["City"] == extract_before_parenthesis(city)]
        # results = results[results["date"] == date]
        # if price_order == "asc":
        #     results = results.sort_values(by=["Average Cost"], ascending=True)
        # elif price_order == "desc":
        #     results = results.sort_values(by=["Average Cost"], ascending=False)

        # if rating_order == "asc":
        #     results = results.sort_values(by=["Aggregate Rating"], ascending=True)
        # elif rating_order == "desc":
        #     results = results.sort_values(by=["Aggregate Rating"], ascending=False)

        return results





# def eat_in_which_city(self, arrives, departs, origin, cities, departure_dates, days):
#         result = []
#         cities = [origin] + cities + [origin]
#         i = 0
#         for day in range(days):
#             if day == departure_dates[i]:
#                 print('hi')
#                 # TODO: not entering???
#                 arrtime = arrives[i]
#                 result.append(If(arrtime > 10, cities[i], cities[i+1])) # do not eat breakfast if arrive time is later than 10
#                 result.append(If(arrtime > 13, cities[i], cities[i+1]))
#                 result.append(If(arrtime > 20, cities[i], cities[i+1]))
#                 # deptime = departs[i+1]
#                 # result.append(If(deptime > 10, cities[i+1], cities[i+2])) # do not eat breakfast if depart time is later than 10
#                 # result.append(If(deptime > 13, cities[i+1], cities[i+2]))
#                 # result.append(If(deptime > 20, cities[i+1], cities[i+2]))
#                 i += 1
#             else:
#                 result.append(cities[i+1])
#                 result.append(cities[i+1])
#                 result.append(cities[i+1])
#         print("Having eat in which info for {} restaurants".format(len(result)))
#         return result


# def run_for_all_cities(self,
#             cities: list,
#             ) -> DataFrame:
#         """Search for flights by origin, destination, and departure date."""
#         results = Array('restaurant', IntSort(), StringSort(), ArraySort(IntSort(), IntSort())) # ori, dest, date, [Price, DepTime, ArrTime], info
#         results_cuisines = Array('restaurant cuisines', IntSort(), StringSort(), ArraySort(IntSort(), StringSort())) # ori, dest, date, [Price, DepTime, ArrTime], info
#         for i, city in enumerate(cities):
#             result = self.data[self.data["City"] == city]
#             if len(result) != 0:
#                 price = Array('Price', IntSort(), IntSort())
#                 cuisines = Array('Cuisines', IntSort(), StringSort())
#                 length = Array('Length', IntSort(), IntSort())
#                 Store(length, 0, len(np.array(result)[:,1]))
#                 for index in range(np.array(result).shape[0]):
#                     import pdb; pdb.set_trace()
#                     Store(price, index, np.array(result)[:,1][index])
#                     Store(cuisines, index, String(np.array(result)[:,2][index]))
#                 Store(results, i, String('Price'), price)
#                 Store(results_cuisines, i, String('Cuisines'), cuisines)
#         return results, results_cuisines