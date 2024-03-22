
all_name_list = ['vase', 'rugby_ball', 'starfish', 'scuba_diver', 'italian_greyhound', 'espresso', 'broccoli', 'cauldron', 'cannon', 'steam_locomotive', 'zebra', 'scarf', 'Granny_Smith', 'tiger', 'panda', 'collie', 'hotdog', 'standard_poodle', 'tennis_ball', 'canoe', 'goldfinch', 'baseball_player', 'yorkshire_terrier', 'stingray', 'gorilla', 'trombone', 'pig', 'hummingbird', 'accordion', 'monarch_butterfly', 'submarine', 'mailbox', 'scottish_terrier', 'bucket', 'eel', 'hyena', 'afghan_hound', 'tractor', 'sea_lion', 'shih_tzu', 'great_white_shark', 'bloodhound', 'school_bus', 'husky', 'bee', 'orangutan', 'timber_wolf', 'puffer_fish', 'baboon', 'pelican', 'flamingo', 'ladybug', 'polar_bear', 'bathtub', 'mitten', 'cocker_spaniels', 'centipede', 'mushroom', 'carousel', 'bagel', 'newt', 'guillotine', 'harp', 'pembroke_welsh_corgi', 'whippet', 'binoculars', 'dalmatian', 'beer_glass', 'boxer', 'backpack', 'grand_piano', 'meerkat', 'missile', 'hermit_crab', 'cowboy_hat', 'ice_cream', 'hammer', 'king_penguin', 'space_shuttle', 'volcano', 'skunk', 'pug', 'axolotl', 'African_chameleon', 'dragonfly', 'red_fox', 'beagle', 'cucumber', 'black_swan', 'junco', 'hen', 'jeep', 'bison', 'birdhouse', 'hatchet', 'strawberry', 'grasshopper', 'grey_whale', 'pineapple', 'boston_terrier', 'burrito', 'saxophone', 'pretzel', 'ostrich', 'lorikeet', 'beaver', 'ant', 'fly', 'guinea_pig', 'gazelle', 'chow_chow', 'bell_pepper', 'labrador_retriever', 'koala', 'fire_engine', 'violin', 'flute', 'chihuahua', 'pirate_ship', 'french_bulldog', 'spider_web', 'banana', 'lawn_mower', 'tree_frog', 'sandal', 'hippopotamus', 'jellyfish', 'cheeseburger', 'electric_guitar', 'toy_poodle', 'bald_eagle', 'lion', 'clown_fish', 'castle', 'candle', 'pomeranian', 'pomegranate', 'chimpanzee', 'parachute', 'rottweiler', 'lemon', 'badger', 'harmonica', 'snow_leopard', 'cabbage', 'iguana', 'wine_bottle', 'mantis', 'military_aircraft', 'cockroach', 'soccer_ball', 'leopard', 'german_shepherd_dog', 'assault_rifle', 'duck', 'west_highland_white_terrier', 'wheelbarrow', 'joystick', 'cobra', 'cheetah', 'scorpion', 'lighthouse', 'killer_whale', 'fox_squirrel', 'pizza', 'golden_retriever', 'saint_bernard', 'lipstick', 'revolver', 'basketball', 'american_egret', 'acorn', 'peacock', 'ambulance', 'toucan', 'lab_coat', 'goldfish', 'barn', 'pickup_truck', 'broom', 'mobile_phone', 'snail', 'border_collie', 'bow_tie', 'hammerhead', 'vulture', 'tabby_cat', 'goose', 'llama', 'shield', 'tarantula', 'schooner', 'tank', 'porcupine', 'gibbon', 'weimaraner', 'basset_hound', 'wood_rabbit', 'lobster', 'gasmask']


sults = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/imgr20_20_3e6_batch2_epoch2_order3_2000exp_re/'
path_results = '/home/ubuntu/code/GMM_camera/GMM/results/imgr20_20_3e6_batch2_epoch2_order3_2000exp_re/'

initial = 20
increment = 20
task_num = 10

import torch
import clip
from PIL import Image
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

all_name = os.listdir(path_results)
all_name.sort()
all_name = sorted(all_name, key=lambda x: int(x.split('.')[0][5:]))
all_mean  = []

for task_id, name in enumerate(all_name):
    with open(path_results + name) as f:
        label_list = []
        msg_list = []
        lines = f.readlines()
        inner_task = [[] for i in range(task_num)]
        idx = 0
        for line in lines:
            if line.startswith('the label is'):
                line = line[13:]
                line = line.strip('\n')
                for tasks in range(task_id + 1):
                    if tasks == 0:
                        if line in all_name_list[:initial]:
                            inner_task[tasks].append(idx)
                    elif line in all_name_list[initial + (tasks-1)*increment: initial+tasks*increment]:
                        inner_task[tasks].append(idx)
                idx += 1             
                label_list.append(line)
            if line.startswith('msg:'):
                line = line[26:]
                line = line.strip('\n')
                line = line.strip('.')
                line = line.strip('#')
                msg_list.append(line) 
        label_set = list(set(label_list))
        text = clip.tokenize(label_set).to(device)

        new_msg_list = []
        for mm in msg_list:
            if len(mm) > 76:
                new_msg_list.append(mm[:76])
            else:
                new_msg_list.append(mm)

        msg_list = new_msg_list

        predict_text = clip.tokenize(msg_list).to(device)

        with torch.no_grad():  
            text_features_label = model.encode_text(text)
            predict_feature = model.encode_text(predict_text)

            text_features_label = text_features_label / text_features_label.norm(dim=1, keepdim=True)
            predict_feature = predict_feature / predict_feature.norm(dim=1, keepdim=True)

        sim  = predict_feature @ text_features_label.T
        real_label = torch.ones(len(msg_list))
        for i in range(len(msg_list)):
            real_label[i] = label_set.index(label_list[i])

        try:
            pre = sim.cpu().argmax(dim=1)
        except:
            import pdb; pdb.set_trace()

        acc = sum(pre == real_label) / len(msg_list)
        task_acc = []
        
        for i in range(task_id + 1):
            task_acc.append(sum(pre[inner_task[i]] == real_label[inner_task[i]])/len(inner_task[i]))

        for acc_each in task_acc:
            print(str(round(float(acc_each*100), 2)), end=' ')

        print( 'mean: ', str(round(float(acc*100), 2)))

        mean = round(float(acc*100), 2)
        all_mean.append(mean)
print('avg: ', sum(all_mean)/len(all_mean))
