import re, string, os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/planner")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../tools/planner")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import importlib
from typing import List, Dict, Any
import tiktoken
from pandas import DataFrame
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from utils.func import load_line_json_data, save_file
import sys
import json
import openai
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
import argparse
from datasets import load_dataset
import os
import pdb
from openai_func import *
import json
from z3 import *
from tools.cities.apis import *
from tools.flights.apis import *
from tools.accommodations.apis import *
from tools.attractions.apis import *
from tools.googleDistanceMatrix.apis import *
from tools.restaurants.apis import *
import time
import pandas

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
# GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']

actionMapping = {"FlightSearch":"flights","AttractionSearch":"attractions","GoogleDistanceMatrix":"googleDistanceMatrix","accommodationSearch":"accommodation","RestaurantSearch":"restaurants","CitySearch":"cities"}


def convert_to_int(real):
    out = ToInt(real) # ToInt(real + 0.0001)
    out += If(real == out, 0, 1)
    return out

def get_arrivals_list(transportation_arrtime, day, variables):
    arrives = []
    if day == 3: 
        arrives.append(transportation_arrtime[0])
        arrives.append(IntVal(-1))
        arrives.append(transportation_arrtime[1])
    elif day == 5:
        arrives.append(transportation_arrtime[0])
        arrives.append(If(variables[1] == 1, transportation_arrtime[1], IntVal(-1)))
        arrives.append(If(variables[1] == 2, transportation_arrtime[1], IntVal(-1)))
        arrives.append(If(variables[1] == 3, transportation_arrtime[1], IntVal(-1)))
        arrives.append(transportation_arrtime[2])
    else:
        arrives.append(transportation_arrtime[0])
        arrives.append(If(variables[1] == 1, transportation_arrtime[1], If(variables[2] == 1, transportation_arrtime[2], IntVal(-1))))
        arrives.append(If(variables[1] == 2, transportation_arrtime[1], If(variables[2] == 2, transportation_arrtime[2], IntVal(-1))))
        arrives.append(If(variables[1] == 3, transportation_arrtime[1], If(variables[2] == 3, transportation_arrtime[2], IntVal(-1))))
        arrives.append(If(variables[1] == 4, transportation_arrtime[1], If(variables[2] == 4, transportation_arrtime[2], IntVal(-1))))
        arrives.append(If(variables[1] == 5, transportation_arrtime[1], If(variables[2] == 5, transportation_arrtime[2], IntVal(-1))))
        arrives.append(transportation_arrtime[3])
    return arrives

def get_city_list(city, day, departure_dates):
    city_list = []
    if day == 3: 
        city_list.append(IntVal(-1))
        city_list.append(IntVal(0))
        city_list.append(IntVal(0))
        city_list.append(IntVal(-1))
    elif day == 5:
        city_list.append(IntVal(-1))
        city_list.append(city[0])
        city_list.append(If(departure_dates[1] <= 1, city[1],city[0]))
        city_list.append(If(departure_dates[1] <= 2, city[1], city[0]))
        city_list.append(If(departure_dates[1] <= 3, city[1], city[0]))
        city_list.append(IntVal(-1))
    else:
        city_list.append(IntVal(-1))
        city_list.append(city[0])
        city_list.append(If(departure_dates[2] <= 1, city[2],If(departure_dates[1] <= 1, city[1],city[0])))
        city_list.append(If(departure_dates[2] <= 2, city[2], If(departure_dates[1] <= 2, city[1],city[0])))
        city_list.append(If(departure_dates[2] <= 3, city[2], If(departure_dates[1] <= 3, city[1],city[0])))
        city_list.append(If(departure_dates[2] <= 4, city[2], If(departure_dates[1] <= 4, city[1],city[0])))
        city_list.append(If(departure_dates[2] <= 5, city[2], If(departure_dates[1] <= 5, city[1],city[0])))
        city_list.append(IntVal(-1))
    
    return city_list

def generate_as_plan(s, variables, query):
    CitySearch = Cities()
    FlightSearch = Flights()
    AttractionSearch = Attractions()
    DistanceSearch = GoogleDistanceMatrix()
    AccommodationSearch = Accommodations()
    RestaurantSearch = Restaurants()
    cities = []
    transportation = []
    departure_dates = []
    transportation_info = []
    restaurant_city_list = []
    attraction_city_list = []
    accommodation_city_list = []
    if query['visiting_city_number'] == 1:
        cities = [query['dest']]
        cities_list = [query['dest']]
    else:
        cities_list = CitySearch.run(query['dest'], query['org'], query['date'])
        if query['org'] in cities_list:
            cities_list.remove(query['org'])
        for city in variables['city']:
            cities.append(cities_list[int(s.model()[city].as_long())])
    for i, flight in enumerate(variables['flight']):
        if bool(s.model()[flight]):
            transportation.append('flight')
        elif bool(s.model()[variables['self-driving'][i]]):
            transportation.append('self-driving')
        else:
            transportation.append('taxi')
    for date_index in variables['departure_dates']:
        departure_dates.append(query['date'][int(s.model()[date_index].as_long())])
    dest_cities = [query['org']] + cities + [query['org']]
    for i, index in enumerate(variables['flight_index']):
        if transportation[i] == 'flight':
            flight_index = int(s.model()[index].as_long())
            flight_list = FlightSearch.run(dest_cities[i], dest_cities[i+1], departure_dates[i])
            # flight_info = f'Flight Number: {np.array(flight_list['Flight Number'])[flight_index]}, from {np.array(flight_list['OriginCityName'])[flight_index]} to {np.array(flight_list['DestCityName'])[flight_index]}, Departure Time: {np.array(flight_list['DepTime'])[flight_index]}, Arrival Time: {np.array(flight_list['ArrTime'])[flight_index]}'
            flight_info = 'Flight Number: {}, from {} to {}, Departure Time: {}, Arrival Time: {}'.format(np.array(flight_list['Flight Number'])[flight_index], np.array(flight_list['OriginCityName'])[flight_index], np.array(flight_list['DestCityName'])[flight_index], np.array(flight_list['DepTime'])[flight_index], np.array(flight_list['ArrTime'])[flight_index])
            transportation_info.append(flight_info)
        elif transportation[i] == 'self-driving':
            transportation_info.append('Self-' + DistanceSearch.run(dest_cities[i], dest_cities[i+1], mode='driving'))
        else:
            # pdb.set_trace()
            transportation_info.append(DistanceSearch.run(dest_cities[i], dest_cities[i+1], mode='taxi'))
    for i,which_city in enumerate(variables['restaurant_in_which_city']):
        # pdb.set_trace()
        city_index = int(s.model()[which_city].as_long())
        if city_index == -1:
            restaurant_city_list.append('-')
        else:
            city = cities_list[city_index]
            restaurant_list = RestaurantSearch.run(city)
            restaurant_index = int(s.model()[variables['restaurant_index'][i]].as_long())
            restaurant = np.array(restaurant_list['Name'])[restaurant_index]
            restaurant_city_list.append(restaurant + ', ' + city)

    for i,which_city in enumerate(variables['attraction_in_which_city']):
        city_index = int(s.model()[which_city].as_long())
        if city_index == -1:
            attraction_city_list.append('-')
        else:
            city = cities_list[city_index]
            attraction_list = AttractionSearch.run(city)
            attraction_index = int(s.model()[variables['attraction_index'][i]].as_long())
            attraction = np.array(attraction_list['Name'])[attraction_index]
            attraction_city_list.append(attraction + ', ' + city)

    for i,city in enumerate(cities):
        accommodation_list = AccommodationSearch.run(city)
        accommodation_index = int(s.model()[variables['accommodation_index'][i]].as_long())
        accommodation = np.array(accommodation_list['NAME'])[accommodation_index]
        accommodation_city_list.append(accommodation + ', ' + city)
    print(cities)
    print(transportation)
    print(departure_dates)
    print(transportation_info)
    print(restaurant_city_list)
    print(attraction_city_list)
    print(accommodation_city_list)
    return f'Destination cities: {cities},\nTransportation dates: {departure_dates},\nTransportation methods between cities: {transportation_info},\nRestaurants (3 meals per day): {restaurant_city_list},\nAttractions (1 per day): {attraction_city_list},\nAccommodations (1 per city): {accommodation_city_list}'

def format_check(code_list, new_code):
    # pdb.set_trace()
    if '\n                ' in code_list[0] and new_code.count('\n                ') <=4:
        new_code = new_code.replace('\n', '\n                ')
    elif '\n            ' in code_list[0]  and new_code.count('\n            ' ) <=4:
        new_code = new_code.replace('\n', '\n            ')
    elif '\n    ' in code_list[0]  and new_code.count('\n    ') <=4:
        new_code = new_code.replace('\n', '\n    ')
    else : 
        new_code = new_code
    new_code = new_code.replace('```python', '')
    new_code = new_code.replace('```', '')
    return new_code

def count_unsat(unsat_cores):
    unsat_reasons = ['valid flight index', 'driving is possible', 'types accomadation', 'budget']
    count = np.zeros(len(unsat_cores))
    reasons = []
    for i, unsat_core in enumerate(unsat_cores):
        single_reason = []
        for j, single_core in enumerate(unsat_core[1:-1].split(",\n")):
            for reason in unsat_reasons:
                if reason in single_core:
                    count[i] += 1
                    single_reason.append(single_core + ' is not satisfied')
        reasons.append(single_reason)
    index = np.argsort(count[count!= 0])
    # pdb.set_trace()
    reason = reasons[index[0]]
    return reason

def interact(query, unsat_cores, code_list, s, suggestions, preference = None):
    FlightSearch = Flights()
    DistanceSearch = GoogleDistanceMatrix()
    AccommodationSearch = Accommodations()
    with open('prompts/interactive_large/info_suggest.txt', 'r') as file:
        info_suggest_prompt = file.read()
    # with open('prompts/interactive_large/info_suggest_nosolver.txt', 'r') as file:
    #     info_suggest_prompt = file.read()
    with open('prompts/interactive_large/code_change.txt', 'r') as file:
        code_change = file.read()
    with open('prompts/interactive_large/json_change.txt', 'r') as file:
        json_change = file.read()
    reason = count_unsat(unsat_cores)
    print("!!!!!!!!!!!!!!!reason: ", reason)
    info_suggest_prompt += str(query) + '\nUnsatisfiable reasons: \n' + str(reason) + '\nCollected information: \n' + '[]'
    # info_suggest_prompt += str(query) + '\nCollected information: \n' + '[]'
    iter = 1
    while True:
        response = GPT_response(info_suggest_prompt, 'gpt-4')
        action = response.split('[')[0]
        print(response)
        # DrivingCheck, DrivingSearch, FlightCheck, FlightSearch, AccommodationSearch, TypeSearch
        if action == 'DrivingCheck':
            city_1 = response.split('[')[1].replace(']','').split(', ')[0].replace('\'','')
            city_2 = response.split('[')[1].replace(']','').split(', ')[1].replace('\'','')
            info = DistanceSearch.run_check(city_1, city_2)#.to_string(index=False)
        elif action == 'DrivingSearch':
            city_1 = response.split('[')[1].replace(']','').replace('\'','')
            info = DistanceSearch.run_search(city_1)
        elif action == 'FlightCheck':
            city_1 = response.split('[')[1].replace(']','').split(', ')[0].replace('\'','')
            city_2 = response.split('[')[1].replace(']','').split(', ')[1].replace('\'','')
            date = response.split('[')[1].replace(']','').split(', ')[2].replace('\'','')
            info = FlightSearch.run_check(city_1, city_2, date)#.to_string(index=False)
        elif action == 'FlightSearch':
            city_1 = response.split('[')[1].replace(']','').split(', ')[0].replace('\'','')
            date = response.split('[')[1].replace(']','').split(', ')[1].replace('\'','')
            info = FlightSearch.run_search(city_1, date)#.to_string(index=False)
        elif action == 'AccommodationSearch':
            city = response.split('[')[1].replace(']','').replace('\'','')
            info = AccommodationSearch.run_search(city)
        elif action == 'TypeSearch':
            category = response.split('[')[1].replace(']','').replace('\'','')
            info = AccommodationSearch.get_type_cities(category)
        elif action == 'Analyze':
            info = response.split('[')[1].replace(']','')
        elif action == 'Suggest':
            suggestion = response.split('[')[1].replace(']', '')
            if preference == 'all_yes': 
                user_input = 'yes'
            else:
                if preference in suggestion:
                    user_input = 'no'
                else:
                    user_input = 'yes'
            if 'destination' in suggestion:
                change_index = 0
            elif 'transportation' in suggestion: 
                change_index = 2
            elif 'house type' in suggestion:
                change_index = 7
            elif 'budget' in suggestion:
                change_index = 8
            else:
                user_input = 'invalid'
                info = 'Your returned actions is not valid. You can only suggest to raise budget, change destination city, remove the house type constraint, or remove the flight/no flight/ no self-driving assertion for transportations'
            if user_input.split(':')[0] == 'yes':
                break 
            elif user_input.split(':')[0] == 'modify':
                suggestion = user_input.split(':')[1]
                break
            elif user_input.split(':')[0] == 'no':
                info = f'You suggested but the user refused to {suggestion} and will not change house {preference} information, please propose other suggestions.\n'
        else:
            info = 'Your returned actions is not valid.'
        if iter >= 15:
            if iter == 16:
                suggestion = 'no change'
                break
            else:
                info += 'Your iteration exceeds 15, you must give a suggestion next action.'
        info_suggest_prompt += '\n-----Iter {}-----'.format(iter) +  '\nAction taken: \n' + response + '\nCollected information: \n' + str(info)
        print(info_suggest_prompt)
        iter += 1
    print(suggestion)
    suggestions.append(suggestion)
    info_suggest_prompt += '\nSuggestion: \n' + suggestion
    if change_index == -1:
        code_change += '\n'.join(code_list) + '\nModified constraints: \n' + suggestion
    else:
        code_change += code_list[change_index] + '\nModified constraints: \n' + suggestion
    json_change += str(query) + '\nModified constraints: \n' + suggestion
    new_code = GPT_response(code_change, 'gpt-4')
    if not 'destination' in suggestion: new_code = format_check(code_list, new_code)
    else: new_code = 'unsat_cores = []\n' + new_code
    code_list[change_index] = new_code
    new_json = GPT_response(json_change, 'gpt-4')
    print(new_code)
    print(new_json)
    return '\n'.join(code_list), code_list, json.loads(new_json), info_suggest_prompt, suggestions


def pipeline(query, mode, user_mode, index):
    path_read =  f'output/{mode}/initial_codes/{index}/'
    path =  f'output/{mode}/{user_mode}/{index}/'
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path+'codes/')
        os.makedirs(path+'plans/')
    
    CitySearch = Cities()
    FlightSearch = Flights()
    AttractionSearch = Attractions()
    DistanceSearch = GoogleDistanceMatrix()
    AccommodationSearch = Accommodations()
    RestaurantSearch = Restaurants()
    s = Optimize()
    variables = {}

    plan = ''
    plan_json = ''
    success = False
    
    query_json = query
    with open(path_read+'codes/' + 'codes.txt', 'r') as f:
        codes = f.read()
    f.close()
    code_list = []
    with open(path_read+'codes/' + 'Destination cities.txt', 'r') as f:
        code_list.append('unsat_cores = []\n'+f.read())
    f.close()
    with open(path_read+'codes/' + 'Departure dates.txt', 'r') as f:
        code_list.append(f.read())
    f.close()
    with open(path_read+'codes/' + 'Transportation methods.txt', 'r') as f:
        code_list.append(f.read())
    f.close()
    with open(path_read+'codes/' + 'Flight information.txt', 'r') as f:
        code_list.append(f.read())
    f.close()
    with open(path_read+'codes/' + 'Driving information.txt', 'r') as f:
        code_list.append(f.read())
    f.close()
    with open(path_read+'codes/' + 'Restaurant information.txt', 'r') as f:
        code_list.append(f.read())
    f.close()
    with open(path_read+'codes/' + 'Attraction information.txt', 'r') as f:
        code_list.append(f.read())
    f.close()
    with open(path_read+'codes/' + 'Accommodation information.txt', 'r') as f:
        code_list.append(f.read())
    f.close()
    with open(path_read+'codes/' + 'Budget.txt', 'r') as f:
        code_list.append(f.read())
    f.close()
    with open('prompts/small/fix/solve_{}.txt'.format(query_json['days']), 'r') as f:
        code_list.append(f.read())
    f.close()

    indent = {3: '\n    ', 5: '\n            ', 7:'\n                '}
    suggestions = []
    for i in range(10):
        local_vars = locals()
        exec(codes, globals(), local_vars)
        if os.path.exists(path+'plans/' + 'plan.txt'):
            print('Found plan')
            break
        codes, code_list, query_json, info_suggest_prompt, suggestions = interact(query_json, local_vars['unsat_cores'], code_list, local_vars['s'], suggestions, preference = user_mode)
        if suggestions[-1] == 'no change' and suggestions[-2] == 'no change' and suggestions[-3] == 'no change':
            raise Exception("Sorry, three consecutive invalid suggestions")
        with open(path+'codes/' + 'codes{}.txt'.format(i), 'w') as f:
            f.write(codes)
        f.close()
        with open(path+'plans/' + 'info_suggest_prompt{}.txt'.format(i), 'w') as f:
            f.write(info_suggest_prompt)
        f.close()
        with open(path+'plans/' + 'json{}.json'.format(i), 'w') as f:
            json.dump(query_json, f)
        f.close()
    with open(path+'plans/' + 'suggestions.txt', 'w') as f:
        f.write(str(suggestions))
    f.close()
    
def collect_inital_codes(query, mode, index, model_version = None):
    path =  f'output/{mode}/initial_codes/{index}/'
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path+'codes/')
        os.makedirs(path+'plans/')
    # setup
    with open('prompts/query_to_json.txt', 'r') as file:
        query_to_json_prompt = file.read()
    with open('prompts/constraint_to_step.txt', 'r') as file:
        constraint_to_step_prompt = file.read()
    with open('prompts/step_to_code_destination_cities.txt', 'r') as file:
        step_to_code_destination_cities_prompt = file.read()
    with open('prompts/step_to_code_departure_dates.txt', 'r') as file:
        step_to_code_departure_dates_prompt = file.read()
    with open('prompts/step_to_code_transportation_methods.txt', 'r') as file:
        step_to_code_transportation_methods_prompt = file.read()
    with open('prompts/step_to_code_flight.txt', 'r') as file:
        step_to_code_flight_prompt = file.read()
    with open('prompts/step_to_code_driving.txt', 'r') as file:
        step_to_code_driving_prompt = file.read()
    with open('prompts/step_to_code_restaurant.txt', 'r') as file:
        step_to_code_restaurant_prompt = file.read()
    with open('prompts/step_to_code_attraction.txt', 'r') as file:
        step_to_code_attraction_prompt = file.read()
    with open('prompts/step_to_code_accommodation.txt', 'r') as file:
        step_to_code_accommodation_prompt = file.read()
    with open('prompts/step_to_code_budget.txt', 'r') as file:
        step_to_code_budget_prompt = file.read()
        
    CitySearch = Cities()
    FlightSearch = Flights()
    AttractionSearch = Attractions()
    DistanceSearch = GoogleDistanceMatrix()
    AccommodationSearch = Accommodations()
    RestaurantSearch = Restaurants()
    s = Optimize()
    variables = {}

    step_to_code_prompts = {'Destination cities': step_to_code_destination_cities_prompt, 
                            'Departure dates': step_to_code_departure_dates_prompt,
                            'Transportation methods': step_to_code_transportation_methods_prompt,
                            'Flight information': step_to_code_flight_prompt,
                            'Driving information': step_to_code_driving_prompt,
                            'Restaurant information': step_to_code_restaurant_prompt,
                            'Attraction information': step_to_code_attraction_prompt,
                            'Accommodation information': step_to_code_accommodation_prompt,
                            'Budget': step_to_code_budget_prompt
                            }
    plan = ''
    plan_json = ''
    success = False
    
    query_json = query
    with open(path+'plans/' + 'query.json', 'w') as f:
      json.dump(query_json, f)
    f.close()

    print('-----------------query in json format-----------------\n',query_json)

    steps = GPT_response(constraint_to_step_prompt + str(query_json) + '\n' + 'Steps:\n', model_version)

    with open(path+'plans/' + 'steps.txt', 'w') as f:
      f.write(steps)
    f.close()

    steps = steps.split('\n\n')
    try:
        codes = 'unsat_cores = []\n'
        for step in steps:
            print('!!!!!!!!!!STEP!!!!!!!!!!\n', step, '\n')
            lines = step.split('# \n')[1]
            prompt = ''
            step_key = ''
            for key in step_to_code_prompts.keys():
                if key in step.split('# \n')[0]:
                    print('!!!!!!!!!!KEY!!!!!!!!!!\n', key, '\n')
                    prompt = step_to_code_prompts[key]
                    step_key = key
            code = GPT_response(prompt + lines, model_version)
            code = code.replace('```python', '')
            code = code.replace('```', '')
            if step_key != 'Destination cities': 
                if query_json['days'] == 3:
                    code = code.replace('\n', '\n    ')
                elif query_json['days'] == 5:
                    code = code.replace('\n', '\n            ')
                else:
                    code = code.replace('\n', '\n                ')
            print('!!!!!!!!!!CODE!!!!!!!!!!\n', code, '\n')
            codes += code + '\n'
            with open(path+'codes/' + f'{step_key}.txt', 'w') as f:
                f.write(code)
            f.close()
        with open('prompts/small/fix/solve_{}.txt'.format(query_json['days']), 'r') as f:
            codes += f.read()
        f.close()
        local_vars = locals()
        exec(codes, globals(), local_vars)
        with open(path+'codes/' + 'codes.txt', 'w') as f:
            f.write(codes)
        f.close()
    except Exception as e:
        with open(path+'codes/' + 'codes.txt', 'w') as f:
            f.write(codes)
        f.close()
        with open(path+'plans/' + 'error.txt', 'w') as f:
            f.write(str(e))
        f.close()

def run_code(mode, user_mode, index):
    path =  f'output/{mode}/{user_mode}/{index}/'
    # path =  f'output/{mode}/{index}/'        
    CitySearch = Cities()
    FlightSearch = Flights()
    AttractionSearch = Attractions()
    DistanceSearch = GoogleDistanceMatrix()
    AccommodationSearch = Accommodations()
    RestaurantSearch = Restaurants()
    s = Optimize()
    variables = {}
    success = False

    with open(path+'plans/' + 'query.json', 'r') as f:
      query_json = json.loads(f.read())
    f.close()

    print('-----------------query in json format-----------------\n',query_json)

    with open(path+'codes/' + 'codes.txt', 'r') as f:
      codes = f.read()
    f.close()
    local_vars = locals()
    exec(codes, globals(), local_vars)
    pdb.set_trace()


if __name__ == '__main__':

    tools_list = ["flights","attractions","accommodations","restaurants","googleDistanceMatrix","cities"]
    csvFile = pandas.read_csv('travelplanner_unsat.csv')
    for mode in ['type', 'all_yes', 'transportation', 'budget', 'destination']:
        for i in range(len(csvFile)):
            try:
                plan_json = json.loads(csvFile.iloc[i]['query'].replace("\'", "\"").replace("None", "null"))
                # First collect inital codes 
                collect_inital_codes(plan_json, 'interactive_large', i+1, 'gpt-4o')
                # Then do the plan repair 
                pipeline(plan_json, 'interactive_large', mode, i+1)
            except Exception as e:
                path =  f'output/interactive_large/{mode}/{i+1}/'
                with open(path+'plans/' + 'error.txt', 'w') as f:
                    f.write(str(e))
                f.close()
