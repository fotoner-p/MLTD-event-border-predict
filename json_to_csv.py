import json
import pandas as pd
from typing import List

event_type: List[str] = ["theater", "tour", "anniversary"]

extract_border: List[int] = [
    100,
    2500,
    5000,
    10000,
    25000,
    50000
]

idol_border: List[int] = [
    10,
    100,
    1000
]

for cur_type in event_type[:2]:
    with open("./%s/event_list.json" % cur_type, "r", encoding="utf-8") as file:
        event_json = json.load(file)

    for event in event_json:
        if event['id'] < 33:
            continue

        rank_list: List[List[int]] = [[] for _ in range(len(extract_border))]

        with open("./%s/%d.json" % (cur_type, event['id']), "r", encoding="utf-8") as file:
            border_json = json.load(file)

        for i in range(len(border_json[0]['data'])):
            for j in range(len(extract_border)):
                try:
                    rank_list[j].append(int(border_json[j]['data'][i]['score']))
                except IndexError:
                    rank_list[j].insert(0, 0)

        rank_result = {}

        for i, cur_list in enumerate(rank_list):
            rank_result['#' + str(extract_border[i])] = cur_list

        df = pd.DataFrame(rank_result)
        df.to_csv("./%s/%d.csv" % (cur_type, event['id']), index=False)

anniversary_id = [44, 92, 142]
anniversary_column = ('1_10', '1_100', '1_1000', '2_10', '2_100', '2_1000', '3_10', '3_100', '3_1000')
rank_list: List[List[int]] = [[] for _ in range(len(anniversary_column))]

for idol_idx in range(1, 53):
    anniversary_list = []

    for a_id in anniversary_id:
        with open("./anniversary/%d_%d.json" % (a_id, idol_idx), "r", encoding="utf-8") as file:
            anniversary_list.append(json.load(file))

    for i, cur_list in enumerate(anniversary_list):
        for j in range(len(idol_border)):
            rank_list[i * 3 + j].append(int(cur_list[j]['data'][-1]["score"]))

rank_result = {}
for i, cur_list in enumerate(rank_list):
    rank_result[anniversary_column[i]] = cur_list

df = pd.DataFrame(rank_result)
df.to_csv("./anniversary/all_idol_value.csv", index=False)
