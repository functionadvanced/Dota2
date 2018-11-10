import requests
import json
import torch
import time
import os
from get_onematch import get_charts

promatch_raw = requests.get("https://api.opendota.com/api/proMatches")
promatch = json.loads(promatch_raw.content)

hero_save_file = r"C:\Users\Ziyu Gong\Desktop\Hackathon\hero"
chart_save_file = r"C:\Users\Ziyu Gong\Desktop\Hackathon\chart"
kill_save_file = r"C:\Users\Ziyu Gong\Desktop\Hackathon\kill"
death_save_file = r"C:\Users\Ziyu Gong\Desktop\Hackathon\death"

if (os.path.isdir(hero_save_file) == False):
    os.mkdir(hero_save_file)
if (os.path.isdir(chart_save_file) == False):
    os.mkdir(chart_save_file)
if (os.path.isdir(kill_save_file) == False):
    os.mkdir(kill_save_file)
if (os.path.isdir(death_save_file) == False):
    os.mkdir(death_save_file)

idx = 0

# print(len(promatch))

for i in range(3, len(promatch)):
    match_id = promatch[i]['match_id']
    match_api = "https://api.opendota.com/api/matches/" + str(match_id)
    print(match_id)
    match_data_raw = requests.get(match_api)
    while (match_data_raw.status_code != 200):
        print("waiting...")
        time.sleep(5)
    time.sleep(2)
    match_data = json.loads(match_data_raw.content)
    state, chart, kill, death, hero = get_charts(match_data)
    if (state == True):
        print("Success")
        torch.save(hero, os.path.join(hero_save_file, str(idx)+".pt"))
        torch.save(chart, os.path.join(chart_save_file, str(idx)+".pt"))
        torch.save(kill, os.path.join(kill_save_file, str(idx)+".pt"))
        torch.save(death, os.path.join(death_save_file, str(idx)+".pt"))
        idx += 1
    else:
        print("Fail")
