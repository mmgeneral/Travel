import pandas as pd
from pandas import DataFrame
from typing import Optional
from utils.func import extract_before_parenthesis
from z3 import *
import numpy as np

class Accommodations:
    def __init__(self, path="database/accommodations/clean_accommodations_2022.csv"):
        self.path = path
        self.data = pd.read_csv(self.path).dropna()[['NAME','price','room type', 'house_rules', 'minimum nights', 'maximum occupancy', 'review rate number', 'city']]
        print("Accommodations loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path).dropna()
    
    def run_search(self, city):
        results = self.data[self.data["city"] == city]
        if len(results) == 0:
            return "There is no accommodation in this city."
        
        return np.unique(results['room type'].to_numpy())
    
    def get_type_cities(self,
            type: str,
            ) -> DataFrame:
        """Search for Accommodations by city and date."""
        # "entire room", "private room", "shared room", "not shared room"
        if type == 'shared room': type = 'Shared room'
        elif type == 'entire room': type = 'Entire home/apt'
        elif type == 'private room': type = 'Private room'
        else: return f"Your input {type} is not valid. Please search for 'entire room', 'private room', or 'shared room'"
        results = self.data[self.data["room type"] == type]
        # the results should show the index
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return f"There is no {type} in all cities."
        return np.unique(results['city'].to_numpy())

    def run(self,
            city: str,
            ) -> DataFrame:
        """Search for accommodations by city."""
        results = self.data[self.data["city"] == city]
        if len(results) == 0:
            return "There is no attraction in this city."
        
        return results
    

    def run_for_all_cities(self, all_cities: list,
            cities: list,
            ) -> DataFrame:
        """Search for flights by origin, destination, and departure date."""
        types_rule_list = ['Private room', 'Entire home/apt', 'Shared room', 'No visitors', 'No smoking', 'No parties', 'No children under 10', 'No pets']
        results = Array('accommodations', IntSort(), IntSort(), ArraySort(IntSort(), IntSort())) # ori, dest, date, [Price, DepTime, ArrTime], info
        # results_hard_constraint = Array('accommodations hard constraint', IntSort(), StringSort(), ArraySort(IntSort(), IntSort(), BoolSort())) # ori, dest, date, [Price, DepTime, ArrTime], info
        results_hard_constraint = Array('accommodations hard constraint', IntSort(), IntSort(), ArraySort(IntSort(), IntSort(), BoolSort())) # ori, dest, date, [Price, DepTime, ArrTime], info
        for i, city in enumerate(cities):
            result = self.data[self.data["city"] == city]
            if len(result) != 0:
                # print('accommodations',city, len(result), len(np.array(result)[:,1]), np.array(result)[:,2], np.array(result)[:,3])
                # print('accommodations',city)                
                # import pdb; pdb.set_trace()
                price = Array('Price', IntSort(), IntSort())
                minimum_nights = Array('Minimum_nights', IntSort(), IntSort())
                maximum_occupancy = Array('Maximum_occupancy', IntSort(), IntSort())
                room_types = Array('Room_types', IntSort(), IntSort(), BoolSort())
                house_rules = Array('House_rules', IntSort(), IntSort(), BoolSort())
                length = Array('Length', IntSort(), IntSort())
                length = Store(length, 0, len(np.array(result)[:,1]))
                # import pdb; pdb.set_trace()
                for index in range(np.array(result).shape[0]):
                    if np.array(result)[:,1][index] is not np.nan:
                        price = Store(price, index, np.array(result)[:,1][index])
                    else:
                        price = 0 #TODO
                    if np.array(result)[:,4][index] is not np.nan:
                        minimum_nights = Store(minimum_nights, index, np.array(result)[:,4][index])
                    else:
                        minimum_nights = Store(minimum_nights, index, 0)
                    if np.array(result)[:,5][index] is not np.nan:
                        maximum_occupancy = Store(maximum_occupancy, index, np.array(result)[:,5][index])
                    else:
                        maximum_occupancy = 10 #TODO
                    room_types_list = np.array(result)[:,2][index]
                    house_rules_list = np.array(result)[:,3][index]
                    # print(room_types_list)
                    # print(house_rules_list)
                    for j in range(3):
                        # print(types_rule_list[j] in room_types_list)
                        room_types = Store(room_types, index, j, types_rule_list[j] in room_types_list)
                    for j in range(3,8):
                        house_rules = Store(house_rules, index, j, types_rule_list[j] in house_rules_list)
                results = Store(results, all_cities.index(city), 0, price)
                results = Store(results, all_cities.index(city), 1, minimum_nights)
                results = Store(results, all_cities.index(city), 2, maximum_occupancy)
                results = Store(results, all_cities.index(city), 3, length)
                results_hard_constraint = Store(results_hard_constraint, all_cities.index(city), 0, room_types)
                results_hard_constraint = Store(results_hard_constraint, all_cities.index(city), 1, house_rules)
            else:
                length = Array('Length', IntSort(), IntSort())
                length = Store(length, 0, -1)
                results = Store(results, all_cities.index(city), 3, length)
        return results, results_hard_constraint

    def get_info(self, info, i, key):
        # ['Price', 'Minimum_nights', 'Maximum_occupancy', 'Length']
        if key == 'Room_types' or key == 'House_rules':
            if key == 'Room_types':
                info_key = Select(info, i, 0)
            else:
                info_key = Select(info, i, 1)
            return info_key, None
        else:
            element = ['Price', 'Minimum_nights', 'Maximum_occupancy', 'Length']
            info_key = Select(info, i, element.index(key))
            info_length = Select(info, i, 3)
            length = Select(info_length, 0)
            return info_key, length
    
    def get_info_for_index(self, price_list, index):
        return Select(price_list, index)

    def check_exists(self, type, accommodation_list, index):
        types_rule_list = ['Private room', 'Entire home/apt', 'Shared room', 'No visitors', 'No smoking', 'No parties', 'No children under 10', 'No pets']
        exists = Select(accommodation_list, index, types_rule_list.index(type))
        return If(index != -1, exists, False)
    
    def run_for_annotation(self,
            city: str,
            ) -> DataFrame:
        """Search for accommodations by city."""
        results = self.data[self.data["city"] == extract_before_parenthesis(city)]
        return results