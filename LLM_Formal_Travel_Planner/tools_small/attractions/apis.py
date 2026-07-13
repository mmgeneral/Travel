import pandas as pd
from pandas import DataFrame
from typing import Optional
from utils.func import extract_before_parenthesis
from z3 import *
import numpy as np

class Attractions:
    def __init__(self, path="database_small/attractions/attractions.csv"):
        self.path = path
        # city,name,review_count,rating,category,attributes,longitude,latitude
        self.data = pd.read_csv(self.path).dropna()[['city','name','review_count','rating','category','attributes',"longitude", "latitude"]]
        # import pdb; pdb.set_trace()
        print("Attractions loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path)

    def get_city_categories(self,
            city: str,
            ) -> DataFrame:
        """Search for Accommodations by city and date."""
        results = self.data[self.data["city"] == city]
        # the results should show the index
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return "There is no attraction in this city."
        return np.unique(results['category'].to_numpy())
    
    def get_category_cities(self,
            category: str,
            ) -> DataFrame:
        """Search for Accommodations by city and date."""
        results = self.data[self.data["category"] == category]
        # the results should show the index
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return f"There is no {category} in all cities."
        return np.unique(results['city'].to_numpy())
    
    def run(self,
            city: str,
            ) -> DataFrame:
        """Search for Accommodations by city and date."""
        results = self.data[self.data["city"] == city]
        # the results should show the index
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return "There is no attraction in this city."
        return results  
    
    def run_for_all_cities(self, all_cities,
            cities: list,
            category_list: list,
            ) -> DataFrame:
        """Search for flights by origin, destination, and departure date."""
        results = Array('attractions', IntSort(), IntSort())
        results_category = Array('attractions category', IntSort(), ArraySort(IntSort(), IntSort(), BoolSort()))
        # category_list = np.unique(self.data['category'].to_numpy())
        if category_list is not None:
            self.category_list = category_list
        else:
            self.category_list = []
        for i, city in enumerate(cities):
            result = self.data[self.data["city"] == city]
            if len(result) != 0:
                # print('attraction', city, len(result), len(np.array(result)[:,1]))
                results = Store(results, all_cities.index(city), IntVal(len(np.array(result)[:,1])))
                category = Array('Category', IntSort(), IntSort(), BoolSort())
                for index in range(np.array(result).shape[0]):
                    for j in range(len(self.category_list)):
                        # if category_list[j] == "Place of worship": import pdb; pdb.set_trace()
                        category = Store(category, index, j, self.category_list[j] == np.array(result)[:,4][index])
                results_category = Store(results_category, all_cities.index(city), category)
            else:
                results = Store(results, all_cities.index(city), -1)
        # import pdb; pdb.set_trace()
        return results, results_category

    def get_info(self, info, i, key):
        if key == 'Category':
            info_key = Select(info, i)
            return info_key
        else:
            length = Select(info, i)
            return length
    
    def get_info_for_index(self, price_list, index):
        return Select(price_list, index)
    
    def check_exists(self, category, category_list, attraction_index):
        categories = self.category_list# list(np.unique(self.data['category'].to_numpy()))
        # import pdb; pdb.set_trace()
        exists = Select(category_list, attraction_index, categories.index(category))
        return If(attraction_index != -1, exists, False)

    # def run_for_annotation(self,
    #         city: str,
    #         ) -> DataFrame:
    #     """Search for Accommodations by city and date."""
    #     results = self.data[self.data["City"] == extract_before_parenthesis(city)]
    #     # the results should show the index
    #     results = results.reset_index(drop=True)
    #     return results