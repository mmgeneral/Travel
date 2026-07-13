import pandas as pd
from pandas import DataFrame
from typing import Optional
from utils.func import extract_before_parenthesis
from z3 import *
import numpy as np
import copy

class Flights:

    def __init__(self, path="database/flights/clean_Flights_2022.csv"):
        self.path = path
        self.data = None

        self.data = pd.read_csv(self.path).dropna()[['Flight Number', 'Price', 'DepTime', 'ArrTime', 'ActualElapsedTime','FlightDate','OriginCityName','DestCityName','Distance']]
        print("Flights API loaded.")
        List = Datatype('List')
        # Constructor cons: (Int, List) -> List
        List.declare('cons', ('car', RealSort()), ('cdr', List))
        # Constructor nil: List
        List.declare('nil')
        # Create the datatype
        self.List = List.create()
        self.cons = self.List.cons
        self.car  = self.List.car
        self.cdr  = self.List.cdr
        self.nil  = self.List.nil

    def load_db(self):
        self.data = pd.read_csv(self.path).dropna().rename(columns={'Unnamed: 0': 'Flight Number'})

    def run_check(self, origin, destination, departure_date):
        results = self.data[self.data["OriginCityName"] == origin]
        results = results[results["DestCityName"] == destination]
        results = results[results["FlightDate"] == departure_date]
        if len(results) == 0:
            return "There is no flight from {} to {} on {}.".format(origin, destination, departure_date)
        else:
            return "Flight exists from {} to {} on {}.".format(origin, destination, departure_date)
    
    def run_search(self, origin, departure_date):
        results = self.data[self.data["OriginCityName"] == origin]
        results = results[results["FlightDate"] == departure_date]
        if len(results) > 0:
            return np.unique(np.array(results['DestCityName']))
        else:
            return "There is no flight from {} on {}.".format(origin, departure_date)
        
    def run(self,
            origin: str,
            destination: str,
            departure_date: str,
            ) -> DataFrame:
        """Search for flights by origin, destination, and departure date."""
        results = self.data[self.data["OriginCityName"] == origin]
        results = results[results["DestCityName"] == destination]
        results = results[results["FlightDate"] == departure_date]
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
            output = []
            for time in times:
                hour = time.split(':')[0]
                minute = time.split(':')[1]
                time_float = int(hour) + float(minute)/60
                output.append(time_float)
            return output
        # cities.append(origin)
        cities = copy.deepcopy(cities_list)
        cities.insert(0, origin)
        all_cities = copy.deepcopy(all_cities)
        all_cities.insert(0, origin)
        results = Array('flights', IntSort(), IntSort(), IntSort(), IntSort(), ArraySort(IntSort(), RealSort())) # ori, dest, date, [Price, DepTime, ArrTime], info
        for i, ori in enumerate(cities):
            if i!= len(cities)-1:
                destination = cities[i+1]
            else:
                destination = cities[0]
            for d, departure_date in enumerate(departure_dates):
                # print(origin, destination)
                result = self.data[self.data["OriginCityName"] == ori]
                # print(result)
                result = result[result["DestCityName"] == destination]
                result = result[result["FlightDate"] == departure_date]
                if len(result) != 0:
                    price = Array('Price', IntSort(), RealSort())
                    depTime = Array('DepTime', IntSort(), RealSort())
                    arrTime = Array('ArrTime', IntSort(), RealSort())
                    length = Array('Length', IntSort(), RealSort())
                    DepTime = convert_time(np.array(result)[:,2])
                    ArrTime = convert_time(np.array(result)[:,3])
                    length = Store(length, 0, len(np.array(result)[:,1]))
                    for index in range(np.array(result).shape[0]):
                        price = Store(price, index, np.array(result)[:,1][index])
                        depTime = Store(depTime, index, DepTime[index])
                        arrTime = Store(arrTime, index, ArrTime[index])
                    results = Store(results, all_cities.index(ori), all_cities.index(destination), d, 0, price)
                    results = Store(results, all_cities.index(ori), all_cities.index(destination), d, 1, depTime)
                    results = Store(results, all_cities.index(ori), all_cities.index(destination), d, 2, arrTime)
                    results = Store(results, all_cities.index(ori), all_cities.index(destination), d, 3, length)
                else:
                    # import pdb; pdb.set_trace()
                    length = Array('Length', IntSort(), RealSort())
                    length = Store(length, 0, -1)
                    results = Store(results, all_cities.index(ori), all_cities.index(destination), d, 3, length)
        return results
    

    def get_info(self, info, i, j, d, key):
        element = ['Price', 'DepTime', 'ArrTime', 'Length']
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
        info_key = Select(info, i, j, d, element.index(key))
        info_length = Select(info, i, j, d, 3)
        length = Select(info_length, 0)
        return info_key, length

    def get_info_for_index(self, price_list, index):
        return Select(price_list, index)
    
    def run_for_annotation(self,
            origin: str,
            destination: str,
            departure_date: str,
            ) -> DataFrame:
        """Search for flights by origin, destination, and departure date."""
        results = self.data[self.data["OriginCityName"] == extract_before_parenthesis(origin)]
        results = results[results["DestCityName"] == extract_before_parenthesis(destination)]
        results = results[results["FlightDate"] == departure_date]
        # if order == "ascPrice":
        #     results = results.sort_values(by=["Price"], ascending=True)
        # elif order == "descPrice":
        #     results = results.sort_values(by=["Price"], ascending=False)
        # elif order == "ascDepTime":
        #     results = results.sort_values(by=["DepTime"], ascending=True)
        # elif order == "descDepTime":
        #     results = results.sort_values(by=["DepTime"], ascending=False)
        # elif order == "ascArrTime":
        #     results = results.sort_values(by=["ArrTime"], ascending=True)
        # elif order == "descArrTime":
        #     results = results.sort_values(by=["ArrTime"], ascending=False)
        return results.to_string(index=False)

    def get_city_set(self):
        city_set = set()
        for unit in self.data['data']:
            city_set.add(unit[5])
            city_set.add(unit[6])





    # def run_for_all_cities_and_dates(self,
    #         origin: str,
    #         cities: list,
    #         departure_dates: list,
    #         ) -> DataFrame:
    #     """Search for flights by origin, destination, and departure date."""
    #     def construct(values):
    #         if len(values) == 1:
    #             return self.cons(values[0], self.nil)
    #         else:
    #             return self.cons(values[0], construct(values[1:]))
        
    #     def convert_time(times):
    #         output = []
    #         for time in times:
    #             hour = time.split(':')[0]
    #             minute = time.split(':')[1]
    #             # output.append(int(hour) + float(minute)/60)
    #             time_float = int(hour) + float(minute)/60
    #             if time_float <= 8:
    #                 output.append(3)
    #             elif time_float <= 12:
    #                 output.append(2)
    #             elif time_float <= 20:
    #                 output.append(1)
    #             else:
    #                 output.append(0)
    #         return output
    #     cities.append(origin)
    #     results = Array('flights', IntSort(), IntSort(), IntSort(), StringSort(), self.List) # ori, dest, date, [Price, DepTime, ArrTime], info
    #     for i, origin in enumerate(cities):
    #         for j, destination in enumerate(cities):
    #             for d, departure_date in enumerate(departure_dates):
    #                 result = self.data[self.data["OriginCityName"] == origin]
    #                 result = result[result["DestCityName"] == destination]
    #                 result = result[result["FlightDate"] == departure_date]
    #                 if len(result) != 0:
    #                     Store(results, i, j, d, String('Price'), construct(np.array(result)[:,1]))
    #                     DepTime = construct(convert_time(np.array(result)[:,2]))
    #                     ArrTime = construct(convert_time(np.array(result)[:,3]))
    #                     Store(results, i, j, d, String('DepTime'), DepTime)
    #                     Store(results, i, j, d, String('ArrTime'), ArrTime)
    #     return results
    
    # def get_info(self, info, i, j, d, key):
    #     info = Select(info, i, j, d, key)
    #     info_copy = copy.deepcopy(info)
    #     length = 0
    #     import pdb; pdb.set_trace()
    #     while simplify(info_copy != self.nil) == True:# returning false???
    #         print("hi!!!!!")
    #         import pdb; pdb.set_trace()
    #         info_copy = simplify(self.cdr(info_copy))
    #         length += 1 
    #     import pdb; pdb.set_trace()
    #     return info, length

    # def get_price(self, price_list, index):
    #     print('start')
    #     price_list_copy = copy.deepcopy(price_list)
    #     length = 0
    #     # import pdb; pdb.set_trace()
    #     while simplify(price_list_copy != self.nil) == True: # returning false????
    #         print('hi')
    #         price_list = self.car(price_list_copy)
    #         price_list_copy = If(IntVal(length) == index, self.nil, self.cdr(price_list_copy))
    #         length += 1
    #     import pdb; pdb.set_trace()
    #     return price_list