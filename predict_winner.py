import torch
import torch.nn as nn
import torch.autograd.variable as Variable
import os
import json

class NnDotaWinner:
    def __init__(self):
        self.batch_size = 20
        self.learning_rate = 5e-4
        self.epochs = 1000

        self.model = nn.Sequential(
            nn.Conv1d(1, 1, 3, stride=3),
            nn.ReLU(),
            nn.Linear(121, 64),
            nn.ReLU(),
            nn.Linear(64,10),
            nn.Sigmoid(),
            )

        self.label = torch.tensor([], dtype = torch.float)
        self.data = torch.tensor([], dtype = torch.float)

        hero_save_file = r"C:\Users\Ziyu Gong\Desktop\Hackathon\hero"
        chart_save_file = r"C:\Users\Ziyu Gong\Desktop\Hackathon\chart"
        kill_save_file = r"C:\Users\Ziyu Gong\Desktop\Hackathon\kill"
        death_save_file = r"C:\Users\Ziyu Gong\Desktop\Hackathon\death"

        for i in range(50):
            idx = torch.load(os.path.join(hero_save_file, '{}.pt'.format(i)))
            chart = torch.load(os.path.join(chart_save_file, '{}.pt'.format(i)))
            kill = torch.load(os.path.join(kill_save_file, '{}.pt'.format(i)))
            death = torch.load(os.path.join(death_save_file, '{}.pt'.format(i)))

            self.label = torch.cat((self.label, chart), 0)
            fights = chart.shape[0]

            for j in range(fights):
                temp = torch.zeros((121*3,1), dtype = torch.float)
                for k in range(10):
                    temp[int(idx[k]*3)] = kill[j, k]
                    temp[int(idx[k]*3+1)] = death[j, k]
                    if (k<4.5):
                        temp[int(idx[k]*3+2)] = 1
                    else:
                        temp[int(idx[k]*3+2)] = 2
                self.data = torch.cat((self.data, torch.t(temp)), 0)

        # self.label = self.label
        self.data = self.data.unsqueeze(1)
        # print(self.label.shape)
        # print(self.data.shape)

        self.loss_fn = nn.MSELoss(reduction = 'sum')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self):

        def training(epoch):

            def generate_databatch(idx):
                label = self.label[idx, :]
                data = self.data[idx, :, :]
                return data, label

            train_idx = torch.randperm(self.batch_size*4)
            training_loss = 0.0

            for i in range(4):
                data, target = generate_databatch(train_idx[i*self.batch_size:(i+1)*self.batch_size])
                # print(data.shape, target.shape)
                data, target = Variable(data), Variable(target)

                output = self.model(data)
                # print(output.shape)
                # print(target.shape)

                batch_loss = self.loss_fn(output.squeeze(), target)
                training_loss += batch_loss.data.item()

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            return training_loss

        def validation(epoch):  # validation method

            valid_idx = [i for i in range(self.batch_size*4, self.batch_size*5)]
            def generate_databatch(idx):
                label = self.label[idx, :]
                data = self.data[idx, :, :]
                return data, label

            self.model.eval()  # Sets the module in evaluation mode
            validation_loss = 0  # initialize total validation loss for whole validation dataset
            total_correct = 0  # no of correct classifications

            for i in range(1):
                data, target = generate_databatch(valid_idx[i*self.batch_size:(i+1)*self.batch_size])
                data = Variable(data)
                output = self.model(data)
                batch_loss = self.loss_fn(output.squeeze(), target)  # compute average MSE loss for current batch
                validation_loss += batch_loss.data.item()  # add current batch loss to total loss

                output = output.squeeze()
                # print(output, target)
                output[output>0.5] = 1
                output[output<0.5] = 0
                correct_per_fight = torch.sum(output == target)/self.batch_size
                total_correct += correct_per_fight

            return total_correct, validation_loss

        for i in range(1, self.epochs + 1):
            # Training
            train_loss = training(i)
            acc, valid_loss = validation(i)

            print("Epoch: {}, Training loss: {:.6f}, Valid loss: {:.6f}, acc: {:.2f}".format(i+1, train_loss, valid_loss, acc))

        torch.save({'state_dict': self.model.state_dict()}, r"C:\Users\Ziyu Gong\Desktop\Hackathon\save.ckp")


    def forward(self, input):
        ckp = torch.load(r"C:\Users\Ziyu Gong\Desktop\Hackathon\save.ckp")
        self.model.load_state_dict(ckp['state_dict'])
        map_idx_raw = '{"antimage": 1, "axe": 2, "bane": 3, "bloodseeker": 4, "crystal_maiden": 5, "drow_ranger": 6, "earthshaker": 7, "juggernaut": 8, "mirana": 9, "morphling": 10, "nevermore": 11, "phantom_lancer": 12, "puck": 13, "pudge": 14, "razor": 15, "sand_king": 16, "storm_spirit": 17, "sven": 18, "tiny": 19, "vengefulspirit": 20, "windrunner": 21, "zuus": 22, "kunkka": 23, "lina": 25, "lion": 26, "shadow_shaman": 27, "slardar": 28, "tidehunter": 29, "witch_doctor": 30, "lich": 31, "riki": 32, "enigma": 33, "tinker": 34, "sniper": 35, "necrolyte": 36, "warlock": 37, "beastmaster": 38, "queenofpain": 39, "venomancer": 40, "faceless_void": 41, "skeleton_king": 42, "death_prophet": 43, "phantom_assassin": 44, "pugna": 45, "templar_assassin": 46, "viper": 47, "luna": 48, "dragon_knight": 49, "dazzle": 50, "rattletrap": 51, "leshrac": 52, "furion": 53, "life_stealer": 54, "dark_seer": 55, "clinkz": 56, "omniknight": 57, "enchantress": 58, "huskar": 59, "night_stalker": 60, "broodmother": 61, "bounty_hunter": 62, "weaver": 63, "jakiro": 64, "batrider": 65, "chen": 66, "spectre": 67, "ancient_apparition": 68, "doom_bringer": 69, "ursa": 70, "spirit_breaker": 71, "gyrocopter": 72, "alchemist": 73, "invoker": 74, "silencer": 75, "obsidian_destroyer": 76, "lycan": 77, "brewmaster": 78, "shadow_demon": 79, "lone_druid": 80, "chaos_knight": 81, "meepo": 82, "treant": 83, "ogre_magi": 84, "undying": 85, "rubick": 86, "disruptor": 87, "nyx_assassin": 88, "naga_siren": 89, "keeper_of_the_light": 90, "wisp": 91, "visage": 92, "slark": 93, "medusa": 94, "troll_warlord": 95, "centaur": 96, "magnataur": 97, "shredder": 98, "bristleback": 99, "tusk": 100, "skywrath_mage": 101, "abaddon": 102, "elder_titan": 103, "legion_commander": 104, "techies": 105, "ember_spirit": 106, "earth_spirit": 107, "abyssal_underlord": 108, "terrorblade": 109, "phoenix": 110, "oracle": 111, "winter_wyvern": 112, "arc_warden": 113, "monkey_king": 114, "dark_willow": 119, "pangolier": 120, "grimstroke": 121}'
        map_idx = json.loads(map_idx_raw)
        data = torch.zeros(121*3, dtype = torch.float)
        for i in range(10):
            hero_id = map_idx[input['hero_list'][i]]
            death = input['hero_death'][input['hero_list'][i]]
            kill = input['hero_kill'][input['hero_list'][i]]
            data[int(hero_id*3)] = kill
            data[int(hero_id*3+1)] = death
            if (i<4.5):
                data[int(hero_id*3+2)] = 1
            else:
                data[int(hero_id*3+2)] = 2
        output = self.model(data.unsqueeze(0).unsqueeze(1)).squeeze()
        return (1-output).tolist()
pre = NnDotaWinner();
# pre.train()
input = {"hero_list": ["shadow_shaman", "storm_spirit", "terrorblade", "earthshaker", "brewmaster", "crystal_maiden", "ursa", "pudge", "invoker", "monkey_king"], "hero_name": ["Shadow Shaman", "Storm Spirit", "Terrorblade", "Earthshaker", "Brewmaster", "Crystal Maiden", "Ursa", "Pudge", "Invoker", "Monkey King"], "hero_kill": {"shadow_shaman": 1, "storm_spirit": 11, "terrorblade": 4, "earthshaker": 6, "brewmaster": 6, "crystal_maiden": 2, "ursa": 1, "pudge": 3, "invoker": 5, "monkey_king": 2}, "hero_death": {"shadow_shaman": 4, "storm_spirit": 2, "terrorblade": 1, "earthshaker": 3, "brewmaster": 4, "crystal_maiden": 9, "ursa": 7, "pudge": 4, "invoker": 4, "monkey_king": 7}}
vec = pre.forward(input)
print(vec)
