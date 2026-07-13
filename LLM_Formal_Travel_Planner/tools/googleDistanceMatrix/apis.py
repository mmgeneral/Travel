import requests
from utils.func import extract_before_parenthesis
import os
from requests.exceptions import SSLError
import time
import sys
import pandas as pd
import numpy as np
from z3 import *
import copy
import math

class GoogleDistanceMatrix:
    def __init__(self, subscription_key: str="") -> None:
        self.gplaces_api_key: str = subscription_key
        self.data =  pd.read_csv('database/googleDistanceMatrix/distance.csv')
        print("GoogleDistanceMatrix loaded.")

    def run_check(self, origin, destination):
        response = self.data[(self.data['origin'] == origin) & (self.data['destination'] == destination)]
        if len(response) > 0:
            if response['duration'].values[0] is None or response['distance'].values[0] is None or response['duration'].values[0] is np.nan or response['distance'].values[0] is np.nan:
                    return f'Driving is not feasible from {origin} to {destination}'
            else:
                return f'Driving exists from {origin} to {destination}'
        else:
            return f'Driving is not feasible from {origin} to {destination}'
    
    def run_search(self, origin):
        response = self.data[(self.data['origin'] == origin)]
        if len(response) > 0:
            return np.unique(np.array(response['destination']))
        else:
            return f'Driving is not feasible from {origin}'
    
    def run(self, origin, destination, mode='driving'):
        origin = extract_before_parenthesis(origin)
        destination = extract_before_parenthesis(destination)
        info = {"origin": origin, "destination": destination,"cost": None, "duration": None, "distance": None}
        response = self.data[(self.data['origin'] == origin) & (self.data['destination'] == destination)]
        if len(response) > 0:
                if response['duration'].values[0] is None or response['distance'].values[0] is None or response['duration'].values[0] is np.nan or response['distance'].values[0] is np.nan:
                    return "No valid information."
                info["duration"] = response['duration'].values[0]
                info["distance"] = response['distance'].values[0]
                if 'driving' in mode:
                    info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")) * 0.05)
                elif mode == "taxi":
                    info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")))
                if 'day' in info["duration"]:
                    return "No valid information."
                return f"{mode}, from {origin} to {destination}, duration: {info['duration']}, distance: {info['distance']}, cost: {info['cost']}"

        return f"{mode}, from {origin} to {destination}, no valid information."   
    
    def run_for_evaluation(self, origin, destination, mode='driving'):
        origin = extract_before_parenthesis(origin)
        destination = extract_before_parenthesis(destination)
        info = {"origin": origin, "destination": destination,"cost": None, "duration": None, "distance": None}
        response = self.data[(self.data['origin'] == origin) & (self.data['destination'] == destination)]
        if len(response) > 0:
                if response['duration'].values[0] is None or response['distance'].values[0] is None or response['duration'].values[0] is np.nan or response['distance'].values[0] is np.nan:
                    return info
                info["duration"] = response['duration'].values[0]
                info["distance"] = response['distance'].values[0]

                if 'day' not in info["duration"]:
                    if 'driving' in mode:
                        info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")) * 0.05)
                    elif mode == "taxi":
                        info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")))

                return info

        return info 


    def run_online(self, origin, destination, mode="driving"):
        # mode in ['driving','taxi','walking', 'distance','transit']
        endpoint = "https://maps.googleapis.com/maps/api/distancematrix/json"

        params = {
            "origins": origin,
            "destinations": destination,
            "mode": mode if mode=="taxi" else "driving",
            "key": self.gplaces_api_key
        }

        while True:
            try:
                response = requests.get(endpoint, params=params)
                break
            except SSLError:
                time.sleep(30)

        data = response.json()
        info = {"origin": origin, "destination": destination,"cost": None, "duration": None, "distance": None}
        if data['status'] == "OK":
            element = data['rows'][0]['elements'][0]
            if element['status'] == "OK":
                info["duration"] = element['duration']['text']
                info["distance"] = element['distance']['text']
                if 'driving' in mode:
                    info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")) * 0.05)
                elif mode == "taxi":
                    info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")))
                # if 'day' in info["duration"]:
                #     return "No valid information."
                return f"{mode}, from {origin} to {destination}, duration: {info['duration']}, distance: {info['distance']}, cost: {info['cost']}"

        return "No valid information."   
    
    def run_for_annotation(self, origin, destination, mode="driving"):
        # mode in ['driving','taxi','walking', 'distance','transit']
        endpoint = "https://maps.googleapis.com/maps/api/distancematrix/json"

        params = {
            "origins": extract_before_parenthesis(origin),
            "destinations": extract_before_parenthesis(destination),
            "mode": mode if mode!="taxi" else "driving",
            "key": self.gplaces_api_key
        }
        
        response = requests.get(endpoint, params=params)
        data = response.json()
        info = {}
        if data['status'] == "OK":
            element = data['rows'][0]['elements'][0]
            if element['status'] == "OK":
                info["duration"] = element['duration']['text']
                info["distance"] = element['distance']['text']
                info["cost"] = None
                if 'driving' in mode:
                    info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")) * 0.05)
                elif mode == "taxi":
                    info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")))
        else:
            info = {"duration": "N/A", "distance": "N/A", "cost": "N/A", "Hint":"Please check the input."}
        return info
    
    def run_for_all_cities(self, origin, all_cities, cities_list):
        """Search for flights by origin, destination, and departure date."""
        def convert_time(time):
            # import pdb; pdb.set_trace()
            if 'hours' in time:
                hour = time.split(' hours')[0]
                time = time.split(' hours')[1][1:]
            elif 'hour' in time:
                hour = time.split(' hour')[0]
                time = time.split(' hour')[1][1:]
            else:
                hour = 0
            if 'mins' in time:
                minute = time.split(' mins')[0]
            elif 'min' in time:
                minute = time.split(' min')[0]
            else:
                minute = 0
            # output.append(int(hour) + float(minute)/60)
            return int(hour) + float(minute)/60
            # except:
            #     import pdb; pdb.set_trace()
        def convert_distance(distance):
            return float(distance.replace("km","").replace(",",""))
        cities = copy.deepcopy(cities_list)
        cities.insert(0, origin)
        all_cities = copy.deepcopy(all_cities)
        all_cities.insert(0, origin)
        results = Array('driving', IntSort(), IntSort(), IntSort(), RealSort()) # ori, dest, date, [Price, DepTime, ArrTime], info
        for i, ori in enumerate(cities):
            if i!= len(cities)-1:
                destination = cities[i+1]
            else:
                destination = cities[0]
            result = self.data[self.data["origin"] == ori]
            result = result[result["destination"] == destination]
            if len(result) != 0 and (type(np.array(result)[0][3])==str or not math.isnan(np.array(result)[0][3])) and 'day' not in np.array(result)[0][3]:
                duration = convert_time(np.array(result)[0][3]) # set it to start driving from 6 am?
                distance = convert_distance(np.array(result)[0][4])
                results = Store(results, all_cities.index(ori), all_cities.index(destination), 0, distance)
                results = Store(results, all_cities.index(ori), all_cities.index(destination), 1, duration)
                results = Store(results, all_cities.index(ori), all_cities.index(destination), 2, distance * 0.05)
                results = Store(results, all_cities.index(ori), all_cities.index(destination), 3, 1)
            else:
                results = Store(results, all_cities.index(ori), all_cities.index(destination), 3, -1)
        return results
    
    def get_info(self, info, i, j, key):
        element = ['Distance', 'Duration', 'Price', 'Length']
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
        info_key = Select(info, i, j, element.index(key))
        length = Select(info, i, j, 3)
        return info_key, length