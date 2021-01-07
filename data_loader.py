import os
from typing import List

import requests
import json
import time

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

sleep_time = 1

for cur_type in event_type:
    if not os.path.exists(cur_type):
        os.mkdir(cur_type)

    res = requests.get("https://api.matsurihi.me/mltd/v1/events?type=%s" % cur_type)

    with open("./%s/event_list.json" % cur_type, "w", encoding='utf-8') as file:
        json.dump(res.json(), file, ensure_ascii=False, indent="\t")

    time.sleep(sleep_time)

for cur_type in event_type[:2]:
    with open("./%s/event_list.json" % cur_type, "r", encoding="utf-8") as file:
        event_json = json.load(file)

    extract_border_str = ",".join([str(border) for border in extract_border])

    for event_info in event_json:
        print(event_info['id'], event_info['name'])
        res = requests.get("https://api.matsurihi.me/mltd/v1/events/%d/rankings/logs/eventPoint/%s" % (event_info['id'], extract_border_str))

        with open("./%s/%d.json" % (cur_type, event_info['id']), "w", encoding='utf-8') as file:
            json.dump(res.json(), file, ensure_ascii=False, indent="\t")

        time.sleep(sleep_time)

with open("./anniversary/event_list.json", "r", encoding="utf-8") as file:
    event_json = json.load(file)
idol_border_str = ",".join([str(border) for border in idol_border])

for event_info in event_json:
    print(event_info['id'], event_info['name'])
    for idol_idx in range(1, 53):
        res = requests.get("https://api.matsurihi.me/mltd/v1/events/%d/rankings/logs/idolPoint/%d/%s" % (event_info['id'], idol_idx, idol_border_str))
        res_json = res.json()

        with open("./anniversary/%d_%d.json" % (event_info['id'], idol_idx), "w", encoding='utf-8') as file:
            json.dump(res_json, file, ensure_ascii=False, indent="\t")

        time.sleep(sleep_time)

