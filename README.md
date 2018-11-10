# Dota2

## Useful link
[combat log](https://steamcommunity.com/sharedfiles/filedetails/?id=309868072)

[use python to read log file in real time](https://stackoverflow.com/questions/5419888/reading-from-a-frequently-updated-file)

[python yield](https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do)

[Live status](https://api.opendota.com/api/live)

[combat log from replay file](https://github.com/mdnpascual/Dota2CLS)

## Objective
1. Ban-pick recommendation
2. Success probabilit prediction at critical time (smoke, roshan, etc)
3. Skill combination


{"hero_list": ["shadow_shaman", "storm_spirit", "terrorblade", "earthshaker", "brewmaster", "crystal_maiden", "ursa", "pudge", "invoker", "monkey_king"], "hero_name": ["Shadow Shaman", "Storm Spirit", "Terrorblade", "Earthshaker", "Brewmaster", "Crystal Maiden", "Ursa", "Pudge", "Invoker", "Monkey King"], "hero_kill": {"shadow_shaman": 1, "storm_spirit": 11, "terrorblade": 4, "earthshaker": 6, "brewmaster": 6, "crystal_maiden": 2, "ursa": 1, "pudge": 3, "invoker": 5, "monkey_king": 2}, "hero_death": {"shadow_shaman": 4, "storm_spirit": 2, "terrorblade": 1, "earthshaker": 3, "brewmaster": 4, "crystal_maiden": 9, "ursa": 7, "pudge": 4, "invoker": 4, "monkey_king": 7}}

## Grammar
from predict_winner import NnDotaWinner

input = {"hero_list": ["shadow_shaman", "storm_spirit", "terrorblade", "earthshaker", "brewmaster", "crystal_maiden", "ursa", "pudge", "invoker", "monkey_king"], "hero_name": ["Shadow Shaman", "Storm Spirit", "Terrorblade", "Earthshaker", "Brewmaster", "Crystal Maiden", "Ursa", "Pudge", "Invoker", "Monkey King"], "hero_kill": {"shadow_shaman": 1, "storm_spirit": 11, "terrorblade": 4, "earthshaker": 6, "brewmaster": 6, "crystal_maiden": 2, "ursa": 1, "pudge": 3, "invoker": 5, "monkey_king": 2}, "hero_death": {"shadow_shaman": 4, "storm_spirit": 2, "terrorblade": 1, "earthshaker": 3, "brewmaster": 4, "crystal_maiden": 9, "ursa": 7, "pudge": 4, "invoker": 4, "monkey_king": 7}}

pre = NnDotaWinner();
vec = pre.forward(input)
