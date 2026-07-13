import json
import pdb
import os.path
# from csp.openai_func import *

def collect_plans(mode, model):
    if mode == 'train':
        index = 45
    elif mode == 'validation':
        index = 180
    else:
        index = 1000
    for j in range(index):
        path =  f'output/{mode}/{model}/{j+1}/plans/'
        if os.path.exists(path + 'plan.txt'):
            # print(j)
            with open(path + 'plan.txt', 'r') as file:
                plan = file.read()
            with open(path + 'query.json', 'r') as file:
                query = json.loads(file.read())
            if plan != '':
                # print(j)
                # try:
                plan_json = [{},{},{},{},{},{},{}]
                plans = plan.split('\n')
                destinations = eval(plans[0][20:-1])
                destinations.append(query['org'])
                destinations.insert(0, query['org'])
                dates = eval(plans[1][22:-1])
                transportation = eval(plans[2][39:-1])
                restaurant = eval(plans[3][31:-1])
                attraction = eval(plans[4][25:-1])
                accommodation = eval(plans[5][29:])
                accommodation.append("-")
                # pdb.set_trace()
                city_index = 0
                try:
                    for i in range(7):
                        if i+1 <= query['days']:
                            plan_json[i]["days"] = i+1
                            if dates[city_index] == query['date'][i]:
                                plan_json[i]["current_city"] = 'from ' + destinations[city_index].replace("\'", "!!!!@@@") + ' to ' + destinations[city_index+1].replace("\'", "!!!!@@@")
                                plan_json[i]["transportation"] = transportation[city_index].replace("\'", "!!!!@@@")
                                city_index += 1
                            else:
                                plan_json[i]["current_city"] = destinations[city_index].replace("\'", "!!!!@@@")
                                plan_json[i]["transportation"] = "-"
                            plan_json[i]["breakfast"] = restaurant[3*i].replace("\'", "!!!!@@@")
                            plan_json[i]["lunch"] = restaurant[3*i + 1].replace("\'", "!!!!@@@")
                            plan_json[i]["dinner"] = restaurant[3*i + 2].replace("\'", "!!!!@@@")
                            plan_json[i]["attraction"] = attraction[i].replace("\'", "!!!!@@@")
                            plan_json[i]["accommodation"] = accommodation[city_index-1].replace("\'", "!!!!@@@").replace('\"', "!@##@!")
                except:
                    print(j+1)
                    plan_json = ''
            else:
                plan_json = ''
        else:
            plan_json = ''
        with open(path+ 'plan_json.txt', 'w') as f:
            f.write(str(plan_json)[1:-1].replace("\'", "\"").replace("!!!!@@@", "\'"))

    f.close()

if __name__ == '__main__':
    collect_plans('train', 'gpt_nl')