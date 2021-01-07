import os

import requests
import json
import time

if not os.path.exists('./cards'):
    os.mkdir('./cards')

sleep_time = 1


for idol_idx in range(1, 53):
    print(idol_idx)
    res = requests.get("https://api.matsurihi.me/mltd/v1/cards?idolId=%d&extraType=pst" % idol_idx)
    res_json = res.json()

    with open("./cards/%d.json" % idol_idx, "w", encoding='utf-8') as file:
        json.dump(res_json, file, ensure_ascii=False, indent="\t")

    time.sleep(sleep_time)


rank_list = {}

for idol_idx in range(1, 53):
    with open("./cards/%d.json" % idol_idx, "r", encoding="utf-8") as file:
        res_json = json.load(file)

    for card in res_json:
        if card['eventId'] not in rank_list:
            rank_list[card['eventId']] = {}

        if card['extraType'] == 2:
            rank_list[card['eventId']]['ranking'] = card['idolId']

        if card['extraType'] == 3:
            rank_list[card['eventId']]['point'] = card['idolId']

with open("./cards/event.json", "w", encoding='utf-8') as file:
    json.dump(rank_list, file, ensure_ascii=False, indent="\t")
