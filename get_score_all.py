# with open('a40_pretrained_exp_batched.txt', 'r') as f:
# with open('newnew_data.txt', 'r') as f:
# with open('break_up4_continual_10_10_distill_weight.txt', 'r') as f:

# with open('./results/img_100_a40_10_10/task10_re.txt', 'r') as f:\

# all_name_list = ['sidewinder rattlesnake', 'kingsnake', 'tick', 'hen', 'vulture', 'peafowl', 'lorikeet', 'sea snake', 'European garden spider', 'tarantula', 'Carolina anole', 'sulphur-crested cockatoo', 'tiger shark', 'bee eater', 'smooth green snake', 'ostrich', 'smooth newt', 'black grouse', 'frilled-necked lizard', 'banded gecko', 'water snake', 'harvestman', 'wolf spider', 'goldfish', 'quail', 'chickadee', 'jay', 'American alligator', 'spotted salamander', 'ring-necked snake', 'junco', 'ptarmigan', 'Gila monster', 'ruffed grouse', 'stingray', 'vine snake', 'prairie grouse', 'bulbul', 'American robin', 'alligator lizard', 'coucal', 'desert grassland whiptail lizard', 'yellow garden spider', 'night snake', 'centipede', 'worm snake', 'American dipper', 'brambling', 'tree frog', 'eastern hog-nosed snake', 'box turtle', 'jacamar', 'indigo bunting', 'scorpion', 'toucan', 'red-breasted merganser', 'duck', 'great white shark', 'green mamba', 'Saharan horned viper', 'agama', 'bald eagle', 'mud turtle', 'partridge', 'great grey owl', 'leatherback sea turtle', 'african grey parrot', 'kite (bird of prey)', 'goose', 'tench', 'macaw', 'newt', 'magpie', 'hummingbird', 'goldfinch', 'house finch', 'chameleon', 'fire salamander', 'American bullfrog', 'European green lizard', 'African rock python', 'trilobite', 'terrapin', 'boa constrictor', 'rooster', 'Indian cobra', 'southern black widow', 'electric ray', 'tailed frog', 'hammerhead shark', 'triceratops', 'Komodo dragon', 'barn spider', 'hornbill', 'green iguana', 'eastern diamondback rattlesnake', 'axolotl', 'Nile crocodile', 'garter snake', 'loggerhead sea turtle']

# all_name_list = ['caterpillar', 'sweet_pepper', 'orchid', 'otter', 'beaver', 'dolphin', 'butterfly', 'baby', 'whale', 'boy', 'mountain', 'turtle', 'clock', 'shrew', 'shark', 'pine_tree', 'elephant', 'possum', 'woman', 'lion', 'house', 'seal', 'pickup_truck', 'tiger', 'skyscraper', 'motorcycle', 'worm', 'ray', 'beetle', 'oak_tree', 'aquarium_fish', 'crocodile', 'mushroom', 'palm_tree', 'mouse', 'chair', 'raccoon', 'cloud', 'trout', 'man', 'cattle', 'sea', 'keyboard', 'bus', 'crab', 'porcupine', 'squirrel', 'wolf', 'bridge', 'sunflower', 'skunk', 'bed', 'apple', 'bee', 'cockroach', 'telephone', 'pear', 'orange', 'couch', 'castle', 'can', 'wardrobe', 'streetcar', 'lizard', 'lobster', 'bear', 'train', 'spider', 'cup', 'kangaroo', 'television', 'hamster', 'forest', 'chimpanzee', 'poppy', 'bottle', 'willow_tree', 'bicycle', 'rose', 'camel', 'rocket', 'bowl', 'rabbit', 'dinosaur', 'girl', 'lamp', 'maple_tree', 'plate', 'flatfish', 'plain', 'leopard', 'tulip', 'tank', 'snake', 'table', 'road', 'tractor', 'snail', 'fox', 'lawn_mower']
# imagenet-r order1
# all_name_list = ['goldfish', 'great_white_shark', 'hammerhead', 'stingray', 'hen', 'ostrich', 'goldfinch', 'junco', 'bald_eagle', 'vulture', 'newt', 'axolotl', 'tree_frog', 'iguana', 'African_chameleon', 'cobra', 'scorpion', 'tarantula', 'centipede', 'peacock', 'lorikeet', 'hummingbird', 'toucan', 'duck', 'goose', 'black_swan', 'koala', 'jellyfish', 'snail', 'lobster', 'hermit_crab', 'flamingo', 'american_egret', 'pelican', 'king_penguin', 'grey_whale', 'killer_whale', 'sea_lion', 'chihuahua', 'shih_tzu', 'afghan_hound', 'basset_hound', 'beagle', 'bloodhound', 'italian_greyhound', 'whippet', 'weimaraner', 'yorkshire_terrier', 'boston_terrier', 'scottish_terrier', 'west_highland_white_terrier', 'golden_retriever', 'labrador_retriever', 'cocker_spaniels', 'collie', 'border_collie', 'rottweiler', 'german_shepherd_dog', 'boxer', 'french_bulldog', 'saint_bernard', 'husky', 'dalmatian', 'pug', 'pomeranian', 'chow_chow', 'pembroke_welsh_corgi', 'toy_poodle', 'standard_poodle', 'timber_wolf', 'hyena', 'red_fox', 'tabby_cat', 'leopard', 'snow_leopard', 'lion', 'tiger', 'cheetah', 'polar_bear', 'meerkat', 'ladybug', 'fly', 'bee', 'ant', 'grasshopper', 'cockroach', 'mantis', 'dragonfly', 'monarch_butterfly', 'starfish', 'wood_rabbit', 'porcupine', 'fox_squirrel', 'beaver', 'guinea_pig', 'zebra', 'pig', 'hippopotamus', 'bison', 'gazelle', 'llama', 'skunk', 'badger', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'baboon', 'panda', 'eel', 'clown_fish', 'puffer_fish', 'accordion', 'ambulance', 'assault_rifle', 'backpack', 'barn', 'wheelbarrow', 'basketball', 'bathtub', 'lighthouse', 'beer_glass', 'binoculars', 'birdhouse', 'bow_tie', 'broom', 'bucket', 'cauldron', 'candle', 'cannon', 'canoe', 'carousel', 'castle', 'mobile_phone', 'cowboy_hat', 'electric_guitar', 'fire_engine', 'flute', 'gasmask', 'grand_piano', 'guillotine', 'hammer', 'harmonica', 'harp', 'hatchet', 'jeep', 'joystick', 'lab_coat', 'lawn_mower', 'lipstick', 'mailbox', 'missile', 'mitten', 'parachute', 'pickup_truck', 'pirate_ship', 'revolver', 'rugby_ball', 'sandal', 'saxophone', 'school_bus', 'schooner', 'shield', 'soccer_ball', 'space_shuttle', 'spider_web', 'steam_locomotive', 'scarf', 'submarine', 'tank', 'tennis_ball', 'tractor', 'trombone', 'vase', 'violin', 'military_aircraft', 'wine_bottle', 'ice_cream', 'bagel', 'pretzel', 'cheeseburger', 'hotdog', 'cabbage', 'broccoli', 'cucumber', 'bell_pepper', 'mushroom', 'Granny_Smith', 'strawberry', 'lemon', 'pineapple', 'banana', 'pomegranate', 'pizza', 'burrito', 'espresso', 'volcano', 'baseball_player', 'scuba_diver', 'acorn']
# imagenet-r order2
# all_name_list = ['banana', 'bucket', 'goldfish', 'barn', 'pineapple', 'ant', 'grand_piano', 'husky', 'hatchet', 'mobile_phone', 'grasshopper', 'fire_engine', 'leopard', 'labrador_retriever', 'cheeseburger', 'flute', 'cannon', 'espresso', 'bagel', 'pomegranate', 'joystick', 'submarine', 'gasmask', 'revolver', 'bathtub', 'lab_coat', 'mitten', 'lorikeet', 'assault_rifle', 'castle', 'carousel', 'jeep', 'king_penguin', 'bald_eagle', 'hammer', 'ice_cream', 'vase', 'bell_pepper', 'ostrich', 'jellyfish', 'fly', 'red_fox', 'tennis_ball', 'american_egret', 'toy_poodle', 'lipstick', 'cabbage', 'sandal', 'dalmatian', 'birdhouse', 'golden_retriever', 'badger', 'yorkshire_terrier', 'ambulance', 'schooner', 'spider_web', 'hermit_crab', 'hen', 'canoe', 'wine_bottle', 'basset_hound', 'broccoli', 'fox_squirrel', 'beer_glass', 'junco', 'llama', 'rugby_ball', 'acorn', 'cockroach', 'goose', 'chow_chow', 'cauldron', 'pretzel', 'mushroom', 'basketball', 'hammerhead', 'boston_terrier', 'backpack', 'whippet', 'flamingo', 'wood_rabbit', 'cheetah', 'pembroke_welsh_corgi', 'lemon', 'volcano', 'great_white_shark', 'bloodhound', 'school_bus', 'orangutan', 'broom', 'vulture', 'tank', 'italian_greyhound', 'scuba_diver', 'baseball_player', 'standard_poodle', 'mantis', 'newt', 'sea_lion', 'parachute', 'timber_wolf', 'chimpanzee', 'cucumber', 'axolotl', 'scottish_terrier', 'candle', 'lighthouse', 'gorilla', 'killer_whale', 'ladybug', 'lawn_mower', 'cobra', 'mailbox', 'saxophone', 'hyena', 'koala', 'soccer_ball', 'polar_bear', 'black_swan', 'strawberry', 'harp', 'monarch_butterfly', 'starfish', 'dragonfly', 'border_collie', 'puffer_fish', 'pig', 'hotdog', 'rottweiler', 'bee', 'german_shepherd_dog', 'tiger', 'beaver', 'hippopotamus', 'afghan_hound', 'lion', 'goldfinch', 'lobster', 'centipede', 'peacock', 'space_shuttle', 'grey_whale', 'pelican', 'toucan', 'guillotine', 'pomeranian', 'boxer', 'bison', 'accordion', 'eel', 'tabby_cat', 'gazelle', 'collie', 'pickup_truck', 'Granny_Smith', 'west_highland_white_terrier', 'tree_frog', 'porcupine', 'clown_fish', 'snow_leopard', 'bow_tie', 'saint_bernard', 'weimaraner', 'meerkat', 'guinea_pig', 'tractor', 'military_aircraft', 'beagle', 'missile', 'chihuahua', 'binoculars', 'scorpion', 'pug', 'electric_guitar', 'shih_tzu', 'cocker_spaniels', 'violin', 'baboon', 'skunk', 'duck', 'zebra', 'gibbon', 'snail', 'iguana', 'steam_locomotive', 'stingray', 'pirate_ship', 'burrito', 'harmonica', 'wheelbarrow', 'panda', 'tarantula', 'scarf', 'cowboy_hat', 'pizza', 'African_chameleon', 'french_bulldog', 'hummingbird', 'trombone', 'shield']
# imagenet-r order3
all_name_list = ['vase', 'rugby_ball', 'starfish', 'scuba_diver', 'italian_greyhound', 'espresso', 'broccoli', 'cauldron', 'cannon', 'steam_locomotive', 'zebra', 'scarf', 'Granny_Smith', 'tiger', 'panda', 'collie', 'hotdog', 'standard_poodle', 'tennis_ball', 'canoe', 'goldfinch', 'baseball_player', 'yorkshire_terrier', 'stingray', 'gorilla', 'trombone', 'pig', 'hummingbird', 'accordion', 'monarch_butterfly', 'submarine', 'mailbox', 'scottish_terrier', 'bucket', 'eel', 'hyena', 'afghan_hound', 'tractor', 'sea_lion', 'shih_tzu', 'great_white_shark', 'bloodhound', 'school_bus', 'husky', 'bee', 'orangutan', 'timber_wolf', 'puffer_fish', 'baboon', 'pelican', 'flamingo', 'ladybug', 'polar_bear', 'bathtub', 'mitten', 'cocker_spaniels', 'centipede', 'mushroom', 'carousel', 'bagel', 'newt', 'guillotine', 'harp', 'pembroke_welsh_corgi', 'whippet', 'binoculars', 'dalmatian', 'beer_glass', 'boxer', 'backpack', 'grand_piano', 'meerkat', 'missile', 'hermit_crab', 'cowboy_hat', 'ice_cream', 'hammer', 'king_penguin', 'space_shuttle', 'volcano', 'skunk', 'pug', 'axolotl', 'African_chameleon', 'dragonfly', 'red_fox', 'beagle', 'cucumber', 'black_swan', 'junco', 'hen', 'jeep', 'bison', 'birdhouse', 'hatchet', 'strawberry', 'grasshopper', 'grey_whale', 'pineapple', 'boston_terrier', 'burrito', 'saxophone', 'pretzel', 'ostrich', 'lorikeet', 'beaver', 'ant', 'fly', 'guinea_pig', 'gazelle', 'chow_chow', 'bell_pepper', 'labrador_retriever', 'koala', 'fire_engine', 'violin', 'flute', 'chihuahua', 'pirate_ship', 'french_bulldog', 'spider_web', 'banana', 'lawn_mower', 'tree_frog', 'sandal', 'hippopotamus', 'jellyfish', 'cheeseburger', 'electric_guitar', 'toy_poodle', 'bald_eagle', 'lion', 'clown_fish', 'castle', 'candle', 'pomeranian', 'pomegranate', 'chimpanzee', 'parachute', 'rottweiler', 'lemon', 'badger', 'harmonica', 'snow_leopard', 'cabbage', 'iguana', 'wine_bottle', 'mantis', 'military_aircraft', 'cockroach', 'soccer_ball', 'leopard', 'german_shepherd_dog', 'assault_rifle', 'duck', 'west_highland_white_terrier', 'wheelbarrow', 'joystick', 'cobra', 'cheetah', 'scorpion', 'lighthouse', 'killer_whale', 'fox_squirrel', 'pizza', 'golden_retriever', 'saint_bernard', 'lipstick', 'revolver', 'basketball', 'american_egret', 'acorn', 'peacock', 'ambulance', 'toucan', 'lab_coat', 'goldfish', 'barn', 'pickup_truck', 'broom', 'mobile_phone', 'snail', 'border_collie', 'bow_tie', 'hammerhead', 'vulture', 'tabby_cat', 'goose', 'llama', 'shield', 'tarantula', 'schooner', 'tank', 'porcupine', 'gibbon', 'weimaraner', 'basset_hound', 'wood_rabbit', 'lobster', 'gasmask']
# all_name_list = ['beer_glass', 'oxcart', 'bearskin', 'drake', 'brass', 'acorn_squash', 'turnstile', 'harvester', 'studio_couch', 'Pomeranian', 'pole', 'cliff_dwelling', 'leaf_beetle', 'titi', 'microphone', 'parachute', 'tobacco_shop', 'bicycle-built-for-two', 'snowplow', 'bassinet', 'Lakeland_terrier', 'flute', 'vacuum', 'jacamar', 'borzoi', 'fire_screen', 'rubber_eraser', 'confectionery', 'tile_roof', 'accordion', 'sidewinder', 'tape_player', 'hand-held_computer', 'school_bus', 'golden_retriever', 'sarong', 'dowitcher', 'ram', 'fireboat', 'birdhouse', 'megalith', 'Italian_greyhound', 'banjo', 'pinwheel', 'Siberian_husky', 'Rottweiler', 'miniature_pinscher', 'swab', 'leafhopper', 'tow_truck', 'sea_snake', 'ice_cream', 'black_and_gold_garden_spider', 'buckeye', 'rocking_chair', 'pelican', 'green_snake', 'English_springer', 'Pekinese', 'patas', 'sleeping_bag', 'vine_snake', 'chain_saw', 'three-toed_sloth', 'beer_bottle', 'Japanese_spaniel', 'king_crab', 'ladle', 'banded_gecko', 'common_newt', 'Norwegian_elkhound', 'rain_barrel', 'balloon', 'walking_stick', 'black-and-tan_coonhound', 'house_finch', 'oxygen_mask', 'acoustic_guitar', 'plate', 'sandbar', 'cock', 'paddlewheel', 'pickup', 'toaster', 'ptarmigan', 'Boston_bull', 'hotdog', 'groom', 'sea_cucumber', 'black-footed_ferret', 'redshank', 'police_van', 'marmoset', 'planetarium', 'cliff', 'totem_pole', 'Great_Pyrenees', 'orange', 'face_powder', 'monarch']
# # all_name_list = ['caterpillar', 'sweet_pepper', 'orchid', 'otter', 'beaver', 'dolphin', 'butterfly', 'baby', 'whale', 'boy', 'mountain', 'turtle', 'clock', 'shrew', 'shark', 'pine_tree', 'elephant', 'possum', 'woman', 'lion', 'house', 'seal', 'pickup_truck', 'tiger', 'skyscraper', 'motorcycle', 'worm', 'ray', 'beetle', 'oak_tree', 'aquarium_fish', 'crocodile', 'mushroom', 'palm_tree', 'mouse', 'chair', 'raccoon', 'cloud', 'trout', 'man', 'cattle', 'sea', 'keyboard', 'bus', 'crab', 'porcupine', 'squirrel', 'wolf', 'bridge', 'sunflower', 'skunk', 'bed', 'apple', 'bee', 'cockroach', 'telephone', 'pear', 'orange', 'couch', 'castle', 'can', 'wardrobe', 'streetcar', 'lizard', 'lobster', 'bear', 'train', 'spider', 'cup', 'kangaroo', 'television', 'hamster', 'forest', 'chimpanzee', 'poppy', 'bottle', 'willow_tree', 'bicycle', 'rose', 'camel', 'rocket', 'bowl', 'rabbit', 'dinosaur', 'girl', 'lamp', 'maple_tree', 'plate', 'flatfish', 'plain', 'leopard', 'tulip', 'tank', 'snake', 'table', 'road', 'tractor', 'snail', 'fox', 'lawn_mower']
# all_name_list = ['tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead', 'electric_ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house_finch', 'junco', 'indigo_bunting', 'robin', 'bulbul', 'jay', 'magpie', 'chickadee', 'water_ouzel', 'kite', 'bald_eagle', 'vulture', 'great_grey_owl', 'European_fire_salamander', 'common_newt', 'eft', 'spotted_salamander', 'axolotl', 'bullfrog', 'tree_frog', 'tailed_frog', 'loggerhead', 'leatherback_turtle', 'mud_turtle', 'terrapin', 'box_turtle', 'banded_gecko', 'common_iguana', 'American_chameleon', 'whiptail', 'agama', 'frilled_lizard', 'alligator_lizard', 'Gila_monster', 'green_lizard', 'African_chameleon', 'Komodo_dragon', 'African_crocodile', 'American_alligator', 'triceratops', 'thunder_snake', 'ringneck_snake', 'hognose_snake', 'green_snake', 'king_snake', 'garter_snake', 'water_snake', 'vine_snake', 'night_snake', 'boa_constrictor', 'rock_python', 'Indian_cobra', 'green_mamba', 'sea_snake', 'horned_viper', 'diamondback', 'sidewinder', 'trilobite', 'harvestman', 'scorpion', 'black_and_gold_garden_spider', 'barn_spider', 'garden_spider', 'black_widow', 'tarantula', 'wolf_spider', 'tick', 'centipede', 'black_grouse', 'ptarmigan', 'ruffed_grouse', 'prairie_chicken', 'peacock', 'quail', 'partridge', 'African_grey', 'macaw', 'sulphur-crested_cockatoo', 'lorikeet', 'coucal', 'bee_eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted_merganser', 'goose']
# # imga
# all_name_list = ['scorpion', 'common_iguana', 'parking_meter', 'schooner', 'lion', 'tarantula', 'basketball', 'mask', 'reel', 'guacamole', 'American_black_bear', 'puffer', 'steam_locomotive', 'wreck', 'Chihuahua', 'sea_lion', 'rapeseed', 'broccoli', 'canoe', 'spider_web', 'bikini', 'hotdog', 'oystercatcher', 'corn', 'marimba', 'flagpole', 'eft', 'puck', 'spatula', 'sundial', 'stingray', 'cello', 'tank', 'cheeseburger', 'bald_eagle', 'leafhopper', 'African_chameleon', 'robin', 'ocarina', 'rhinoceros_beetle', 'vulture', 'pelican', 'chest', "yellow_lady's_slipper", 'grand_piano', 'submarine', 'sleeping_bag', 'sandal', 'envelope', 'bell_pepper', 'sewing_machine', 'fox_squirrel', 'balance_beam', 'kimono', 'banjo', 'bullfrog', 'red_fox', 'beacon', 'mongoose', 'Persian_cat', 'feather_boa', 'go-kart', 'pretzel', 'jay', 'airliner', 'torch', 'fly', 'stethoscope', 'iron', 'suspension_bridge', 'drake', 'junco', 'goldfinch', 'viaduct', 'walking_stick', 'manhole_cover', 'parachute', 'studio_couch', 'mosque', 'cabbage_butterfly', 'forklift', 'doormat', 'American_egret', 'ballplayer', 'chain', 'pool_table', 'harvestman', 'limousine', 'teddy', 'lighter', 'bow_tie', 'ant', 'skunk', 'soap_dispenser', 'acorn', 'goblet', 'bison', 'beaker', 'barn', 'bow', 'goose', 'volcano', 'rugby_ball', 'academic_gown', 'African_elephant', 'mushroom', 'revolver', 'toaster', 'hermit_crab', 'jellyfish', 'organ', 'barrow', 'ladybug', 'snowplow', 'German_shepherd', 'obelisk', 'snowmobile', 'balloon', 'baboon', 'marmot', 'mitten', 'dragonfly', 'fountain', 'piggy_bank', 'jeep', 'apron', 'box_turtle', "jack-o'-lantern", 'dumbbell', 'mantis', 'water_tower', 'volleyball', 'unicycle', 'starfish', 'cowboy_boot', 'cockroach', 'capuchin', 'armadillo', 'custard_apple', 'sulphur-crested_cockatoo', 'agama', 'lemon', 'hand_blower', 'Rottweiler', 'snail', 'porcupine', 'acoustic_guitar', 'flamingo', 'broom', 'banana', 'wood_rabbit', 'nail', 'garter_snake', 'crayfish', 'bee', 'umbrella', 'school_bus', 'rocking_chair', 'washer', 'sax', 'sea_anemone', 'maraca', 'cliff', 'pomegranate', 'golden_retriever', 'American_alligator', 'pug', 'lorikeet', 'ambulance', 'golfcart', 'garbage_truck', 'accordion', 'wine_bottle', 'toucan', 'racket', 'hummingbird', 'centipede', 'koala', 'castle', 'grasshopper', 'monarch', 'Christmas_stocking', 'cradle', 'bubble', 'candle', 'lynx', 'shovel', 'weevil', 'dial_telephone', 'digital_clock', 'lycaenid', 'carbonara', 'breastplate', 'saltshaker', 'flatworm', 'tricycle', 'cucumber', 'drumstick', 'syringe', 'quill']

# tiny
# all_name_list = ['seashore', 'scoreboard', 'plunger', 'chest', 'Persian_cat', 'candle', 'steel_arch_bridge', 'bathtub', 'fur_coat', 'gondola', 'remote_control', 'oboe', 'barrel', 'Egyptian_cat', 'beach_wagon', 'wok', 'pretzel', 'lesser_panda', 'iPod', 'koala', 'cardigan', 'punching_bag', 'albatross', 'abacus', 'snail', 'convertible', 'chimpanzee', 'mantis', 'pomegranate', 'Labrador_retriever', 'jellyfish', 'dumbbell', 'academic_gown', 'wooden_spoon', 'German_shepherd', 'space_heater', 'pill_bottle', 'kimono', 'sea_slug', 'vestment', 'fountain', 'gasmask', 'brain_coral', 'sea_cucumber', 'espresso', 'lawn_mower', 'sombrero', 'sunglasses', 'stopwatch', 'cockroach', 'sandal', 'refrigerator', 'tarantula', 'Christmas_stocking', 'banana', 'American_lobster', 'cougar', 'potpie', 'torch', 'poncho', 'beacon', 'gazelle', 'go-kart', 'black_widow', 'hog', 'sock', 'bighorn', 'monarch', 'sports_car', 'umbrella', 'altar', 'king_penguin', 'cash_machine', 'tractor', 'fly', 'bell_pepper', 'teddy', 'barbershop', 'moving_van', 'European_fire_salamander', 'birdhouse', 'guacamole', 'hourglass', 'bucket', 'orange', 'comic_book', 'bannister', 'backpack', 'dragonfly', 'crane', 'school_bus', 'brown_bear', 'snorkel', 'thatch', 'picket_fence', 'bullfrog', 'drumstick', 'golden_retriever', 'black_stork', 'goldfish', 'lemon', 'alp', 'trilobite', 'dugong', 'grasshopper', 'tabby', 'cliff', 'police_van', 'scorpion', 'pizza', 'meat_loaf', 'basketball', 'boa_constrictor', 'standard_poodle', 'mushroom', 'African_elephant', 'walking_stick', 'teapot', 'water_tower', 'spider_web', 'binoculars', 'cannon', 'bullet_train', 'lifeboat', 'guinea_pig', 'sulphur_butterfly', 'frying_pan', 'pay-phone', 'flagpole', 'acorn', 'ladybug', 'jinrikisha', 'military_uniform', 'freight_car', 'sewing_machine', 'lakeside', 'bison', 'suspension_bridge', 'beer_bottle', 'lion', 'desk', 'parking_meter', 'broom', 'rugby_ball', 'beaker', 'baboon', 'centipede', 'coral_reef', 'miniskirt', 'projectile', 'swimming_trunks', 'confectionery', 'tailed_frog', 'slug', 'dining_table', 'pop_bottle', 'mashed_potato', 'reel', 'Yorkshire_terrier', 'apron', 'cauliflower', 'Chihuahua', 'computer_keyboard', 'goose', 'spiny_lobster', 'dam', 'butcher_shop', 'pole', 'ox', 'volleyball', 'orangutan', 'triumphal_arch', 'bee', 'barn', 'water_jug', 'ice_lolly', 'turnstile', 'trolleybus', 'cliff_dwelling', 'Arabian_camel', 'bow_tie', 'CD_player', 'nail', 'American_alligator', 'lampshade', 'neck_brace', 'syringe', 'viaduct', 'ice_cream', 'rocking_chair', 'obelisk', 'chain', 'brass', 'magnetic_compass', 'maypole', 'limousine', 'bikini', 'plate', "potter's_wheel", 'organ']
# tiny order2
# all_name_list = ['alp', 'chest', 'Chihuahua', 'space_heater', 'trolleybus', 'dining_table', 'neck_brace', 'plunger', 'lion', 'beach_wagon', 'goose', 'potpie', 'abacus', 'koala', 'flagpole', 'brass', 'turnstile', 'cauliflower', 'jinrikisha', 'refrigerator', 'ox', 'dumbbell', 'water_tower', 'candle', 'dragonfly', 'binoculars', 'tabby', 'teapot', 'pill_bottle', 'scorpion', 'volleyball', 'obelisk', 'cockroach', 'Christmas_stocking', 'pole', 'backpack', 'organ', 'nail', 'mashed_potato', 'sulphur_butterfly', 'academic_gown', 'cougar', 'monarch', 'Yorkshire_terrier', 'projectile', 'golden_retriever', 'beer_bottle', 'fly', 'bighorn', 'mantis', 'American_alligator', 'African_elephant', 'barrel', 'pomegranate', 'thatch', 'basketball', 'walking_stick', 'rocking_chair', 'lakeside', 'barn', 'ice_cream', "potter's_wheel", 'pretzel', 'kimono', 'dugong', 'rugby_ball', 'sandal', 'frying_pan', 'spider_web', 'triumphal_arch', 'moving_van', 'trilobite', 'European_fire_salamander', 'Egyptian_cat', 'fountain', 'bannister', 'orange', 'plate', 'sea_cucumber', 'brown_bear', 'swimming_trunks', 'baboon', 'military_uniform', 'slug', 'pizza', 'butcher_shop', 'miniskirt', 'police_van', 'acorn', 'pay-phone', 'lawn_mower', 'coral_reef', 'bison', 'boa_constrictor', 'sewing_machine', 'beaker', 'lesser_panda', 'stopwatch', 'grasshopper', 'gasmask', 'desk', 'tailed_frog', 'sombrero', 'bullet_train', 'American_lobster', 'Arabian_camel', 'pop_bottle', 'banana', 'parking_meter', 'mushroom', 'picket_fence', 'fur_coat', 'bucket', 'comic_book', 'oboe', 'guinea_pig', 'bell_pepper', 'brain_coral', 'snorkel', 'lampshade', 'snail', 'lemon', 'sea_slug', 'CD_player', 'ladybug', 'gazelle', 'beacon', 'limousine', 'bee', 'wok', 'standard_poodle', 'broom', 'black_stork', 'confectionery', 'Persian_cat', 'reel', 'gondola', 'chain', 'syringe', 'centipede', 'tarantula', 'viaduct', 'vestment', 'espresso', 'birdhouse', 'cardigan', 'bow_tie', 'iPod', 'barbershop', 'suspension_bridge', 'German_shepherd', 'bathtub', 'scoreboard', 'cannon', 'hourglass', 'poncho', 'dam', 'torch', 'steel_arch_bridge', 'tractor', 'magnetic_compass', 'spiny_lobster', 'goldfish', 'cliff_dwelling', 'bikini', 'seashore', 'sunglasses', 'Labrador_retriever', 'lifeboat', 'computer_keyboard', 'ice_lolly', 'apron', 'water_jug', 'altar', 'school_bus', 'go-kart', 'teddy', 'jellyfish', 'maypole', 'albatross', 'bullfrog', 'punching_bag', 'sock', 'chimpanzee', 'meat_loaf', 'sports_car', 'orangutan', 'convertible', 'crane', 'cliff', 'black_widow', 'umbrella', 'hog', 'king_penguin', 'cash_machine', 'wooden_spoon', 'guacamole', 'remote_control', 'drumstick', 'freight_car']
# tiny order3
# all_name_list = ['teddy', 'bell_pepper', 'projectile', 'beacon', 'Yorkshire_terrier', 'umbrella', 'turnstile', 'golden_retriever', 'ice_cream', 'cliff_dwelling', 'kimono', 'bighorn', 'cannon', 'poncho', 'cockroach', 'pay-phone', 'cash_machine', 'dugong', 'Persian_cat', 'water_jug', 'espresso', 'oboe', 'chest', 'king_penguin', 'guinea_pig', 'water_tower', 'walking_stick', 'lesser_panda', 'confectionery', 'organ', 'seashore', 'ice_lolly', 'baboon', 'sandal', 'gazelle', 'punching_bag', 'tarantula', 'dumbbell', 'miniskirt', 'syringe', 'fly', 'mantis', 'beaker', 'CD_player', 'guacamole', 'limousine', 'brown_bear', 'orange', 'altar', 'potpie', 'academic_gown', 'thatch', 'sea_slug', 'obelisk', 'pole', 'meat_loaf', 'bullet_train', 'snail', 'pretzel', 'tabby', 'goose', 'comic_book', 'mashed_potato', 'scoreboard', 'space_heater', 'picket_fence', 'apron', 'volleyball', 'scorpion', 'albatross', 'birdhouse', 'jinrikisha', 'refrigerator', 'cardigan', 'plate', 'frying_pan', 'African_elephant', 'trilobite', 'bikini', 'ox', "potter's_wheel", 'flagpole', 'mushroom', 'reel', 'koala', 'gondola', 'police_van', 'wooden_spoon', 'stopwatch', 'rocking_chair', 'lion', 'lawn_mower', 'sulphur_butterfly', 'pill_bottle', 'cougar', 'black_widow', 'desk', 'black_stork', 'rugby_ball', 'hourglass', 'go-kart', 'lakeside', 'dining_table', 'wok', 'chimpanzee', 'lemon', 'computer_keyboard', 'grasshopper', 'boa_constrictor', 'lifeboat', 'dragonfly', 'spiny_lobster', 'Arabian_camel', 'steel_arch_bridge', 'tractor', 'convertible', 'torch', 'pop_bottle', 'snorkel', 'barn', 'sewing_machine', 'sports_car', 'standard_poodle', 'brass', 'spider_web', 'trolleybus', 'beach_wagon', 'magnetic_compass', 'Chihuahua', 'Egyptian_cat', 'plunger', 'alp', 'bison', 'viaduct', 'German_shepherd', 'backpack', 'banana', 'barbershop', 'sea_cucumber', 'European_fire_salamander', 'chain', 'bow_tie', 'orangutan', 'centipede', 'abacus', 'cliff', 'military_uniform', 'pizza', 'Labrador_retriever', 'fountain', 'monarch', 'crane', 'candle', 'triumphal_arch', 'teapot', 'dam', 'American_lobster', 'parking_meter', 'bannister', 'hog', 'drumstick', 'bathtub', 'freight_car', 'beer_bottle', 'swimming_trunks', 'pomegranate', 'bullfrog', 'vestment', 'moving_van', 'binoculars', 'sunglasses', 'iPod', 'suspension_bridge', 'sombrero', 'tailed_frog', 'brain_coral', 'Christmas_stocking', 'jellyfish', 'acorn', 'fur_coat', 'bee', 'remote_control', 'sock', 'coral_reef', 'maypole', 'gasmask', 'slug', 'butcher_shop', 'broom', 'basketball', 'neck_brace', 'goldfish', 'American_alligator', 'school_bus', 'nail', 'bucket', 'barrel', 'cauliflower', 'lampshade', 'ladybug']


path_results = '/home/ubuntu/code/new-mini/MiniGPT-4/results/finetune_10_10_batch2_5_epoch_64_eval/'
# path_results = './results/finetune_cifar100/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/finetune_imgr_20_20/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/finetune_imgr_batch2_epoch1/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/img100_random/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/img100_random_few50/'

path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/test_img100_random_1epoch_lr1e7/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/test_img100_random_all_1e6_6e8_batch8/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/test_img100_random_all_1e6_6e8_batch8_epoch3/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/test_img100_random_all_1e6_6e8_batch8_epoch5/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/test_cifar100_random_all_3e6_1e6_batch2_epoch3/'

path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/test_img100_random_all_3e6_1e6_batch2_epoch3/'

path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/test_cifar100_random_all_3e7_1e7_batch2_epoch3/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/set5_5_cifar100_random_all_3e7_1e7_batch2_epoch3/'

path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/img100_random_few30_3e5_1e5_batch2_epoch2/'
# path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/test_img100_random_all_3e7_1e7_batch2_epoch2/'
# path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/re_cifar100_random_all_3e7_1e7_batch2_epoch3/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/img100_random_few30_3e6_1e6_batch2_epoch2/'

path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/img100_old_few200_3e5_1e5_batch2_epoch2/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/img100_old_few50_3e5_1e5_batch2_epoch2/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/img100_old_few25_3e5_1e5_batch2_epoch2/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/imga_all_3e5_1e5_batch2_epoch1/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/imga_all_3e5_1e5_batch2_epoch5/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/cifar_all_3e6_1e6_batch2_epoch2/'

# path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/cifar_few40_3e5_1e5_batch2_epoch1_wop/'
# path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/cifar5_5_3e6_batch2_epoch2_2000exp/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/imgnet-r/imgr20_20_3e5_batch2_epoch1_few40_order3_500exp/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_5_batch2_epoch2_3e7_all_adaptlr/'
# path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_10_batch2_epoch2_3e7_all_adaptlr/'
# path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_20_batch2_epoch2_3e7_all_adaptlr/'
# path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_5_batch2_epoch2_3e7_all_adaptlr_2000exp/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_5_batch2_epoch2_3e7_all_adaptlr_2000exp_a/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_10_batch2_epoch2_3e7_all_adaptlr_2000exp_a/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_20_batch2_epoch2_3e7_all_adaptlr_2000exp_a/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_10_batch2_epoch2_3e7_all_adaptlr_re_3_inc/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_10_batch2_epoch2_3e6_all_adaptlr_re_1_inc_woexp/'

path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_10_batch2_epoch1_2_3e6_7_order1/'

path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_10_batch2_epoch1_2_3e6_7_order2/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_5_batch2_epoch2_3e6_7_order3/'
# path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_10_batch2_epoch2_3e7_all_adaptlr/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_20_batch2_epoch2_3e6_7_order2_re/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_5_batch2_epoch2_3e6_7_order2_re/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_10_batch3_epoch2_3e6_7_order3_re/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_20_batch2_epoch2_3e6_7_order3_re/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_20_batch2_epoch2_3e6_7_order3_2000_re/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_10_batch2_epoch2_3e6_7_order3_2000_re/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_20_batch2_epoch2_3e6_7_order2_2000_re/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_10_batch2_epoch2_3e6_7_order2_2000_re/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/tiny_100_5_batch2_epoch2_3e6_7_order2_2000_re/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/re_img100_10_10_epoch1_batch2/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/re_img100_10_10_epoch2_batch2/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/re_img100_10_10_epoch5_batch2/'
path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/re_img100_10_10_epoch5_batch3/'

path_results = '/root/siton-gpfs-caoxusheng/code/mini-CIL/results/imgr20_20_3e6_batch2_epoch2_order3_2000exp_re/'
path_results = '/home/ubuntu/code/GMM_camera/GMM/results/imgr20_20_3e6_batch2_epoch2_order3_2000exp_re/'
# initial = 5
# increment = 5
# task_num = 20
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

# import pdb; pdb.set_trace()
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
        
        # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()

        for i in range(len(msg_list)):
            real_label[i] = label_set.index(label_list[i])


        try:
            pre = sim.cpu().argmax(dim=1)
        except:
            import pdb; pdb.set_trace()



        acc = sum(pre == real_label) / len(msg_list)


        task_acc = []

        
        # import pdb;pdb.set_trace()
        for i in range(task_id + 1):
            task_acc.append(sum(pre[inner_task[i]] == real_label[inner_task[i]])/len(inner_task[i]))

        for acc_each in task_acc:
            print(str(round(float(acc_each*100), 2)), end=' ')

        print( 'mean: ', str(round(float(acc*100), 2)))

        mean = round(float(acc*100), 2)
        all_mean.append(mean)
        # print('')
        # import pdb; pdb.set_trace()
print('avg: ', sum(all_mean)/len(all_mean))


# task_acc.append(sum(pre[innn_task1] == real_label[innn_task1])/len(innn_task1))
# task_acc.append(sum(pre[innn_task2] == real_label[innn_task2])/len(innn_task2))
# task_acc.append(sum(pre[innn_task3] == real_label[innn_task3])/len(innn_task3))
# task_acc.append(sum(pre[innn_task4] == real_label[innn_task4])/len(innn_task4))
# task_acc.append(sum(pre[innn_task5] == real_label[innn_task5])/len(innn_task5))
# task_acc.append(sum(pre[innn_task6] == real_label[innn_task6])/len(innn_task6))
# task_acc.append(sum(pre[innn_task7] == real_label[innn_task7])/len(innn_task7))
# task_acc.append(sum(pre[innn_task8] == real_label[innn_task8])/len(innn_task8))
# task_acc.append(sum(pre[innn_task9] == real_label[innn_task9])/len(innn_task9))
# task_acc.append(sum(pre[innn_task10] == real_label[innn_task10])/len(innn_task10))
# print('acc first50: ', acc_first50)

# for i in task_acc:
#     print(str(round(float(i)*100, 2)) + ", ", end=" ")

# print('')
