import json
import pdb
import os.path

def collect_plans(mode, model):
    if mode == 'train':
        index = 45
    elif mode == 'validation':
        index = 180
    else:
        index = 1000
    plan_list = []
    for i in range(index):
        path =  f'output/{mode}/{model}/{i+1}/plans/'
        plan_list_single = []
        if os.path.exists(path + 'plan_json.txt'):
            with open(path + 'plan_json.txt', 'r') as file:
                plan_json = file.read()
                if plan_json == '':
                    with open(path + 'query.txt', 'r') as file:
                        query = file.read()
                    entry = {"idx": i+1, "query": query, "plan": None}
                    plan_list.append(entry)
                else:
                    plan_days = plan_json.split('},')
                    for j, plan_day in enumerate(plan_days):
                        if j == 6:
                            corrected = plan_day.replace('\n', '')
                        else:
                            corrected = plan_day.replace('\n', '') + '}'
                        print(corrected)
                        if not '''{}''' in corrected:
                            print('!!')
                            plan_list_single.append(json.loads(corrected))
                    with open(path + 'query.txt', 'r') as file:
                        query = file.read()
                    entry = {"idx": i+1, "query": query, "plan": plan_list_single}
                    plan_list.append(entry)
    with open(f'output/{mode}_{model}.jsonl', 'w') as outfile:
        for entry in plan_list:
            json.dump(entry, outfile)
            outfile.write('\n')

def check_plans(mode, model):
    if mode == 'train':
        index = 45
    elif mode == 'validation':
        index = 180
    else:
        index = 1000
    plan_list = []
    count = 0
    for i in range(index):
        path =  f'output/{mode}/{model}/{i+1}/plans/'
        if not os.path.exists(path + 'plan.txt'):
            print(i+1)
            count+=1
    print('total', count)


if __name__ == '__main__':
    collect_plans('train', 'gpt_nl')
