from pandas import DataFrame
from tools.accommodations.apis import *
from tools.restaurants.apis import *
from tools.googleDistanceMatrix.apis import *
from tools.flights.apis import *

class Cities:
    def __init__(self ,path="database/background/citySet_with_states.txt") -> None:
        self.path = path
        self.load_data()
        print("Cities loaded.")

    def load_data(self):
        cityStateMapping = open(self.path, "r").read().strip().split("\n")
        self.data = {}
        for unit in cityStateMapping:
            city, state = unit.split("\t")
            if state not in self.data:
                self.data[state] = [city]
            else:
                self.data[state].append(city)
    
    def run(self, state, origin, dates) -> dict:
        if state not in self.data:
            return ValueError("Invalid State")
        # elif state == 'Texas':
        else:
            city_list = self.data[state]
            # print(city_list)
            AccommodationSearch = Accommodations()
            RestaurantSearch = Restaurants()
            FlightSearch = Flights()
            DistanceSearch = GoogleDistanceMatrix()
            good = []
            bad = []
            for city in city_list:
                flight = FlightSearch.run(origin, city, dates[0])
                drive = DistanceSearch.run(origin, city)
                if not 'no flight' in flight or not 'No valid information' in drive:
                    good.append(city)
                else:
                    bad.append(city)
            return good + bad