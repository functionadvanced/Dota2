import torch
import json

def get_charts(match_data):
    map_json = '{"1": "npc_dota_hero_antimage", "2": "npc_dota_hero_axe", "3": "npc_dota_hero_bane", "4": "npc_dota_hero_bloodseeker", "5": "npc_dota_hero_crystal_maiden", "6": "npc_dota_hero_drow_ranger", "7": "npc_dota_hero_earthshaker", "8": "npc_dota_hero_juggernaut", "9": "npc_dota_hero_mirana", "10": "npc_dota_hero_morphling", "11": "npc_dota_hero_nevermore", "12": "npc_dota_hero_phantom_lancer", "13": "npc_dota_hero_puck", "14": "npc_dota_hero_pudge", "15": "npc_dota_hero_razor", "16": "npc_dota_hero_sand_king", "17": "npc_dota_hero_storm_spirit", "18": "npc_dota_hero_sven", "19": "npc_dota_hero_tiny", "20": "npc_dota_hero_vengefulspirit", "21": "npc_dota_hero_windrunner", "22": "npc_dota_hero_zuus", "23": "npc_dota_hero_kunkka", "25": "npc_dota_hero_lina", "26": "npc_dota_hero_lion", "27": "npc_dota_hero_shadow_shaman", "28": "npc_dota_hero_slardar", "29": "npc_dota_hero_tidehunter", "30": "npc_dota_hero_witch_doctor", "31": "npc_dota_hero_lich", "32": "npc_dota_hero_riki", "33": "npc_dota_hero_enigma", "34": "npc_dota_hero_tinker", "35": "npc_dota_hero_sniper", "36": "npc_dota_hero_necrolyte", "37": "npc_dota_hero_warlock", "38": "npc_dota_hero_beastmaster", "39": "npc_dota_hero_queenofpain", "40": "npc_dota_hero_venomancer", "41": "npc_dota_hero_faceless_void", "42": "npc_dota_hero_skeleton_king", "43": "npc_dota_hero_death_prophet", "44": "npc_dota_hero_phantom_assassin", "45": "npc_dota_hero_pugna", "46": "npc_dota_hero_templar_assassin", "47": "npc_dota_hero_viper", "48": "npc_dota_hero_luna", "49": "npc_dota_hero_dragon_knight", "50": "npc_dota_hero_dazzle", "51": "npc_dota_hero_rattletrap", "52": "npc_dota_hero_leshrac", "53": "npc_dota_hero_furion", "54": "npc_dota_hero_life_stealer", "55": "npc_dota_hero_dark_seer", "56": "npc_dota_hero_clinkz", "57": "npc_dota_hero_omniknight", "58": "npc_dota_hero_enchantress", "59": "npc_dota_hero_huskar", "60": "npc_dota_hero_night_stalker", "61": "npc_dota_hero_broodmother", "62": "npc_dota_hero_bounty_hunter", "63": "npc_dota_hero_weaver", "64": "npc_dota_hero_jakiro", "65": "npc_dota_hero_batrider", "66": "npc_dota_hero_chen", "67": "npc_dota_hero_spectre", "68": "npc_dota_hero_ancient_apparition", "69": "npc_dota_hero_doom_bringer", "70": "npc_dota_hero_ursa", "71": "npc_dota_hero_spirit_breaker", "72": "npc_dota_hero_gyrocopter", "73": "npc_dota_hero_alchemist", "74": "npc_dota_hero_invoker", "75": "npc_dota_hero_silencer", "76": "npc_dota_hero_obsidian_destroyer", "77": "npc_dota_hero_lycan", "78": "npc_dota_hero_brewmaster", "79": "npc_dota_hero_shadow_demon", "80": "npc_dota_hero_lone_druid", "81": "npc_dota_hero_chaos_knight", "82": "npc_dota_hero_meepo", "83": "npc_dota_hero_treant", "84": "npc_dota_hero_ogre_magi", "85": "npc_dota_hero_undying", "86": "npc_dota_hero_rubick", "87": "npc_dota_hero_disruptor", "88": "npc_dota_hero_nyx_assassin", "89": "npc_dota_hero_naga_siren", "90": "npc_dota_hero_keeper_of_the_light", "91": "npc_dota_hero_wisp", "92": "npc_dota_hero_visage", "93": "npc_dota_hero_slark", "94": "npc_dota_hero_medusa", "95": "npc_dota_hero_troll_warlord", "96": "npc_dota_hero_centaur", "97": "npc_dota_hero_magnataur", "98": "npc_dota_hero_shredder", "99": "npc_dota_hero_bristleback", "100": "npc_dota_hero_tusk", "101": "npc_dota_hero_skywrath_mage", "102": "npc_dota_hero_abaddon", "103": "npc_dota_hero_elder_titan", "104": "npc_dota_hero_legion_commander", "105": "npc_dota_hero_techies", "106": "npc_dota_hero_ember_spirit", "107": "npc_dota_hero_earth_spirit", "108": "npc_dota_hero_abyssal_underlord", "109": "npc_dota_hero_terrorblade", "110": "npc_dota_hero_phoenix", "111": "npc_dota_hero_oracle", "112": "npc_dota_hero_winter_wyvern", "113": "npc_dota_hero_arc_warden", "114": "npc_dota_hero_monkey_king", "119": "npc_dota_hero_dark_willow", "120": "npc_dota_hero_pangolier", "121": "npc_dota_hero_grimstroke"}'
    map = json.loads(map_json)

    player_idx = {}
    hero_idx_dict = {}
    draft = match_data['draft_timings']
    for i in range(len(draft)):
        if (draft[i]['pick'] == True):
            id = draft[i]['hero_id']
            player_idx[map[str(id)]] = draft[i]['player_slot']
            hero_idx_dict[draft[i]['player_slot']] = id
    hero_idx = torch.zeros(10, dtype=torch.float)
    for i in range(10):
        hero_idx[i] = hero_idx_dict[i]
    print(hero_idx_dict, hero_idx)

    teamfights = match_data['teamfights']
    death_chart = torch.zeros(len(teamfights), 10, dtype=torch.float)
    teamfightstime = torch.zeros(len(teamfights), dtype=torch.float)

    for i in range(len(teamfights)):
        teamfightstime[i] = teamfights[i]['start']
        for j in range(10):
            death_chart[i][j] = teamfights[i]['players'][j]['deaths']
    # print(teamfightstime)
    # print(death_chart)

    kills_log = []
    for i in range(10):
        kills = match_data['players'][i]['kills_log']
        for j in range(len(kills)):
            entry = torch.tensor([i, player_idx[kills[j]['key']], kills[j]['time']])
            kills_log.append(entry)

    kill_data = torch.zeros(len(teamfights), 10, dtype=torch.float)
    death_data = torch.zeros(len(teamfights), 10, dtype=torch.float)
    for i in range(len(teamfights)):
        time = teamfightstime[i]
        for j in range(len(kills_log)):
            if (kills_log[j][2] < time):
                kill_data[i, kills_log[j][0]] = kill_data[i, kills_log[j][0]] + 1
                death_data[i, kills_log[j][1]] = death_data[i, kills_log[j][1]] + 1

    return True, death_chart, kill_data, death_data, hero_idx
