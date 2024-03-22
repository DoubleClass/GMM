import os
import torch.nn as nn

from continuum import ClassIncremental, InstanceIncremental
from continuum.datasets import (
    CIFAR100, ImageNet100, TinyImageNet200, ImageFolderDataset, Core50
)
from .utils import get_dataset_class_names
from torchvision import transforms

class ImageNet1000(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)
    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()


class ImageNet_R(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)
    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "test")
        return super().get_data()


def get_dataset(cfg, is_train, transforms=None):
    if cfg.dataset == "cifar100":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = CIFAR100(
            data_path=data_path, 
            download=True, 
            train=is_train, 
            # transforms=transforms
        )
        classes_names = dataset.dataset.classes

    elif cfg.dataset == "tinyimagenet":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = TinyImageNet200(
            data_path, 
            train=is_train,
            download=True
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
    elif cfg.dataset == "tiny":
        data_path = '/'
        dataset = ImageNet100(
            data_path, 
            train=is_train,
            data_subset=os.path.join('/', "train_200_random_tiny.txt" if is_train else "val_200_random_tiny.txt")
        )
        classes_names = ['seashore', 'scoreboard', 'plunger', 'chest', 'Persian_cat', 'candle', 'steel_arch_bridge', 'bathtub', 'fur_coat', 'gondola', 'remote_control', 'oboe', 'barrel', 'Egyptian_cat', 'beach_wagon', 'wok', 'pretzel', 'lesser_panda', 'iPod', 'koala', 'cardigan', 'punching_bag', 'albatross', 'abacus', 'snail', 'convertible', 'chimpanzee', 'mantis', 'pomegranate', 'Labrador_retriever', 'jellyfish', 'dumbbell', 'academic_gown', 'wooden_spoon', 'German_shepherd', 'space_heater', 'pill_bottle', 'kimono', 'sea_slug', 'vestment', 'fountain', 'gasmask', 'brain_coral', 'sea_cucumber', 'espresso', 'lawn_mower', 'sombrero', 'sunglasses', 'stopwatch', 'cockroach', 'sandal', 'refrigerator', 'tarantula', 'Christmas_stocking', 'banana', 'American_lobster', 'cougar', 'potpie', 'torch', 'poncho', 'beacon', 'gazelle', 'go-kart', 'black_widow', 'hog', 'sock', 'bighorn', 'monarch', 'sports_car', 'umbrella', 'altar', 'king_penguin', 'cash_machine', 'tractor', 'fly', 'bell_pepper', 'teddy', 'barbershop', 'moving_van', 'European_fire_salamander', 'birdhouse', 'guacamole', 'hourglass', 'bucket', 'orange', 'comic_book', 'bannister', 'backpack', 'dragonfly', 'crane', 'school_bus', 'brown_bear', 'snorkel', 'thatch', 'picket_fence', 'bullfrog', 'drumstick', 'golden_retriever', 'black_stork', 'goldfish', 'lemon', 'alp', 'trilobite', 'dugong', 'grasshopper', 'tabby', 'cliff', 'police_van', 'scorpion', 'pizza', 'meat_loaf', 'basketball', 'boa_constrictor', 'standard_poodle', 'mushroom', 'African_elephant', 'walking_stick', 'teapot', 'water_tower', 'spider_web', 'binoculars', 'cannon', 'bullet_train', 'lifeboat', 'guinea_pig', 'sulphur_butterfly', 'frying_pan', 'pay-phone', 'flagpole', 'acorn', 'ladybug', 'jinrikisha', 'military_uniform', 'freight_car', 'sewing_machine', 'lakeside', 'bison', 'suspension_bridge', 'beer_bottle', 'lion', 'desk', 'parking_meter', 'broom', 'rugby_ball', 'beaker', 'baboon', 'centipede', 'coral_reef', 'miniskirt', 'projectile', 'swimming_trunks', 'confectionery', 'tailed_frog', 'slug', 'dining_table', 'pop_bottle', 'mashed_potato', 'reel', 'Yorkshire_terrier', 'apron', 'cauliflower', 'Chihuahua', 'computer_keyboard', 'goose', 'spiny_lobster', 'dam', 'butcher_shop', 'pole', 'ox', 'volleyball', 'orangutan', 'triumphal_arch', 'bee', 'barn', 'water_jug', 'ice_lolly', 'turnstile', 'trolleybus', 'cliff_dwelling', 'Arabian_camel', 'bow_tie', 'CD_player', 'nail', 'American_alligator', 'lampshade', 'neck_brace', 'syringe', 'viaduct', 'ice_cream', 'rocking_chair', 'obelisk', 'chain', 'brass', 'magnetic_compass', 'maypole', 'limousine', 'bikini', 'plate', "potter's_wheel", 'organ']

    elif cfg.dataset == "imagenet100":
        data_path = '/'
        dataset = ImageNet100(
            data_path, 
            train=is_train,
            data_subset=os.path.join('/', "train_100_dytox.txt" if is_train else "val_100_dytox.txt")
        )
        classes_names = get_dataset_class_names()
        classes_names = ['beer_glass', 'oxcart', 'bearskin', 'drake', 'brass', 'acorn_squash', 'turnstile', 'harvester', 'studio_couch', 'Pomeranian', 'pole', 'cliff_dwelling', 'leaf_beetle', 'titi', 'microphone', 'parachute', 'tobacco_shop', 'bicycle-built-for-two', 'snowplow', 'bassinet', 'Lakeland_terrier', 'flute', 'vacuum', 'jacamar', 'borzoi', 'fire_screen', 'rubber_eraser', 'confectionery', 'tile_roof', 'accordion', 'sidewinder', 'tape_player', 'hand-held_computer', 'school_bus', 'golden_retriever', 'sarong', 'dowitcher', 'ram', 'fireboat', 'birdhouse', 'megalith', 'Italian_greyhound', 'banjo', 'pinwheel', 'Siberian_husky', 'Rottweiler', 'miniature_pinscher', 'swab', 'leafhopper', 'tow_truck', 'sea_snake', 'ice_cream', 'black_and_gold_garden_spider', 'buckeye', 'rocking_chair', 'pelican', 'green_snake', 'English_springer', 'Pekinese', 'patas', 'sleeping_bag', 'vine_snake', 'chain_saw', 'three-toed_sloth', 'beer_bottle', 'Japanese_spaniel', 'king_crab', 'ladle', 'banded_gecko', 'common_newt', 'Norwegian_elkhound', 'rain_barrel', 'balloon', 'walking_stick', 'black-and-tan_coonhound', 'house_finch', 'oxygen_mask', 'acoustic_guitar', 'plate', 'sandbar', 'cock', 'paddlewheel', 'pickup', 'toaster', 'ptarmigan', 'Boston_bull', 'hotdog', 'groom', 'sea_cucumber', 'black-footed_ferret', 'redshank', 'police_van', 'marmoset', 'planetarium', 'cliff', 'totem_pole', 'Great_Pyrenees', 'orange', 'face_powder', 'monarch']
        classes_names = ['tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead', 'electric_ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house_finch', 'junco', 'indigo_bunting', 'robin', 'bulbul', 'jay', 'magpie', 'chickadee', 'water_ouzel', 'kite', 'bald_eagle', 'vulture', 'great_grey_owl', 'European_fire_salamander', 'common_newt', 'eft', 'spotted_salamander', 'axolotl', 'bullfrog', 'tree_frog', 'tailed_frog', 'loggerhead', 'leatherback_turtle', 'mud_turtle', 'terrapin', 'box_turtle', 'banded_gecko', 'common_iguana', 'American_chameleon', 'whiptail', 'agama', 'frilled_lizard', 'alligator_lizard', 'Gila_monster', 'green_lizard', 'African_chameleon', 'Komodo_dragon', 'African_crocodile', 'American_alligator', 'triceratops', 'thunder_snake', 'ringneck_snake', 'hognose_snake', 'green_snake', 'king_snake', 'garter_snake', 'water_snake', 'vine_snake', 'night_snake', 'boa_constrictor', 'rock_python', 'Indian_cobra', 'green_mamba', 'sea_snake', 'horned_viper', 'diamondback', 'sidewinder', 'trilobite', 'harvestman', 'scorpion', 'black_and_gold_garden_spider', 'barn_spider', 'garden_spider', 'black_widow', 'tarantula', 'wolf_spider', 'tick', 'centipede', 'black_grouse', 'ptarmigan', 'ruffed_grouse', 'prairie_chicken', 'peacock', 'quail', 'partridge', 'African_grey', 'macaw', 'sulphur-crested_cockatoo', 'lorikeet', 'coucal', 'bee_eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted_merganser', 'goose']

        # import pdb; pdb.set_trace()
    elif cfg.dataset == "imageneta":
        data_path = '/'
        dataset = ImageNet100(
            data_path, 
            train=is_train,
            data_subset=os.path.join('/', "train_100_dytox.txt" if is_train else "val_imga_200.txt")
        )
        imga100 = ['scorpion', 'common_iguana', 'parking_meter', 'schooner', 'lion', 'tarantula', 'basketball', 'mask', 'reel', 'guacamole', 'American_black_bear', 'puffer', 'steam_locomotive', 'wreck', 'Chihuahua', 'sea_lion', 'rapeseed', 'broccoli', 'canoe', 'spider_web', 'bikini', 'hotdog', 'oystercatcher', 'corn', 'marimba', 'flagpole', 'eft', 'puck', 'spatula', 'sundial', 'stingray', 'cello', 'tank', 'cheeseburger', 'bald_eagle', 'leafhopper', 'African_chameleon', 'robin', 'ocarina', 'rhinoceros_beetle', 'vulture', 'pelican', 'chest', "yellow_lady's_slipper", 'grand_piano', 'submarine', 'sleeping_bag', 'sandal', 'envelope', 'bell_pepper', 'sewing_machine', 'fox_squirrel', 'balance_beam', 'kimono', 'banjo', 'bullfrog', 'red_fox', 'beacon', 'mongoose', 'Persian_cat', 'feather_boa', 'go-kart', 'pretzel', 'jay', 'airliner', 'torch', 'fly', 'stethoscope', 'iron', 'suspension_bridge', 'drake', 'junco', 'goldfinch', 'viaduct', 'walking_stick', 'manhole_cover', 'parachute', 'studio_couch', 'mosque', 'cabbage_butterfly', 'forklift', 'doormat', 'American_egret', 'ballplayer', 'chain', 'pool_table', 'harvestman', 'limousine', 'teddy', 'lighter', 'bow_tie', 'ant', 'skunk', 'soap_dispenser', 'acorn', 'goblet', 'bison', 'beaker', 'barn', 'bow', 'goose', 'volcano', 'rugby_ball', 'academic_gown', 'African_elephant', 'mushroom', 'revolver', 'toaster', 'hermit_crab', 'jellyfish', 'organ', 'barrow', 'ladybug', 'snowplow', 'German_shepherd', 'obelisk', 'snowmobile', 'balloon', 'baboon', 'marmot', 'mitten', 'dragonfly', 'fountain', 'piggy_bank', 'jeep', 'apron', 'box_turtle', "jack-o'-lantern", 'dumbbell', 'mantis', 'water_tower', 'volleyball', 'unicycle', 'starfish', 'cowboy_boot', 'cockroach', 'capuchin', 'armadillo', 'custard_apple', 'sulphur-crested_cockatoo', 'agama', 'lemon', 'hand_blower', 'Rottweiler', 'snail', 'porcupine', 'acoustic_guitar', 'flamingo', 'broom', 'banana', 'wood_rabbit', 'nail', 'garter_snake', 'crayfish', 'bee', 'umbrella', 'school_bus', 'rocking_chair', 'washer', 'sax', 'sea_anemone', 'maraca', 'cliff', 'pomegranate', 'golden_retriever', 'American_alligator', 'pug', 'lorikeet', 'ambulance', 'golfcart', 'garbage_truck', 'accordion', 'wine_bottle', 'toucan', 'racket', 'hummingbird', 'centipede', 'koala', 'castle', 'grasshopper', 'monarch', 'Christmas_stocking', 'cradle', 'bubble', 'candle', 'lynx', 'shovel', 'weevil', 'dial_telephone', 'digital_clock', 'lycaenid', 'carbonara', 'breastplate', 'saltshaker', 'flatworm', 'tricycle', 'cucumber', 'drumstick', 'syringe', 'quill']

        classes_names = imga100
        # import pdb; pdb.set_trace()

    elif cfg.dataset == "imagenet1000":
        data_path = '/'
        dataset = ImageNet1000(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "imagenet_R":
        data_path = '/path/to/imagenet-r/'
        dataset = ImageNet_R(
            data_path, 
            train=is_train
        )
        classes_names = ['goldfish', 'great_white_shark', 'hammerhead', 'stingray', 'hen', 'ostrich', 'goldfinch', 'junco', 'bald_eagle', 'vulture', 'newt', 'axolotl', 'tree_frog', 'iguana', 'African_chameleon', 'cobra', 'scorpion', 'tarantula', 'centipede', 'peacock', 'lorikeet', 'hummingbird', 'toucan', 'duck', 'goose', 'black_swan', 'koala', 'jellyfish', 'snail', 'lobster', 'hermit_crab', 'flamingo', 'american_egret', 'pelican', 'king_penguin', 'grey_whale', 'killer_whale', 'sea_lion', 'chihuahua', 'shih_tzu', 'afghan_hound', 'basset_hound', 'beagle', 'bloodhound', 'italian_greyhound', 'whippet', 'weimaraner', 'yorkshire_terrier', 'boston_terrier', 'scottish_terrier', 'west_highland_white_terrier', 'golden_retriever', 'labrador_retriever', 'cocker_spaniels', 'collie', 'border_collie', 'rottweiler', 'german_shepherd_dog', 'boxer', 'french_bulldog', 'saint_bernard', 'husky', 'dalmatian', 'pug', 'pomeranian', 'chow_chow', 'pembroke_welsh_corgi', 'toy_poodle', 'standard_poodle', 'timber_wolf', 'hyena', 'red_fox', 'tabby_cat', 'leopard', 'snow_leopard', 'lion', 'tiger', 'cheetah', 'polar_bear', 'meerkat', 'ladybug', 'fly', 'bee', 'ant', 'grasshopper', 'cockroach', 'mantis', 'dragonfly', 'monarch_butterfly', 'starfish', 'wood_rabbit', 'porcupine', 'fox_squirrel', 'beaver', 'guinea_pig', 'zebra', 'pig', 'hippopotamus', 'bison', 'gazelle', 'llama', 'skunk', 'badger', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'baboon', 'panda', 'eel', 'clown_fish', 'puffer_fish', 'accordion', 'ambulance', 'assault_rifle', 'backpack', 'barn', 'wheelbarrow', 'basketball', 'bathtub', 'lighthouse', 'beer_glass', 'binoculars', 'birdhouse', 'bow_tie', 'broom', 'bucket', 'cauldron', 'candle', 'cannon', 'canoe', 'carousel', 'castle', 'mobile_phone', 'cowboy_hat', 'electric_guitar', 'fire_engine', 'flute', 'gasmask', 'grand_piano', 'guillotine', 'hammer', 'harmonica', 'harp', 'hatchet', 'jeep', 'joystick', 'lab_coat', 'lawn_mower', 'lipstick', 'mailbox', 'missile', 'mitten', 'parachute', 'pickup_truck', 'pirate_ship', 'revolver', 'rugby_ball', 'sandal', 'saxophone', 'school_bus', 'schooner', 'shield', 'soccer_ball', 'space_shuttle', 'spider_web', 'steam_locomotive', 'scarf', 'submarine', 'tank', 'tennis_ball', 'tractor', 'trombone', 'vase', 'violin', 'military_aircraft', 'wine_bottle', 'ice_cream', 'bagel', 'pretzel', 'cheeseburger', 'hotdog', 'cabbage', 'broccoli', 'cucumber', 'bell_pepper', 'mushroom', 'Granny_Smith', 'strawberry', 'lemon', 'pineapple', 'banana', 'pomegranate', 'pizza', 'burrito', 'espresso', 'volcano', 'baseball_player', 'scuba_diver', 'acorn']


    elif cfg.dataset == "core50":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = dataset = Core50(
            data_path, 
            scenario="domains", 
            classification="category", 
            train=is_train
        )
        classes_names = [
            "plug adapters", "mobile phones", "scissors", "light bulbs", "cans", 
            "glasses", "balls", "markers", "cups", "remote controls"
        ]
    
    else:
        ValueError(f"'{cfg.dataset}' is a invalid dataset.")

    return dataset, classes_names


def build_cl_scenarios(cfg, is_train, transforms) -> nn.Module:

    dataset, classes_names = get_dataset(cfg, is_train)
    # import pdb; pdb.set_trace()
    if cfg.scenario == "class":
        scenario = ClassIncremental(
            dataset,
            initial_increment=cfg.initial_increment,
            increment=cfg.increment,
            transformations=transforms.transforms, # Convert Compose into list
            class_order=cfg.class_order,
        )

    elif cfg.scenario == "domain":
        scenario = InstanceIncremental(
            dataset,
            transformations=transforms.transforms,
        )

    elif cfg.scenario == "task-agnostic":
        NotImplementedError("Method has not been implemented. Soon be added.")

    else:
        ValueError(f"You have entered `{cfg.scenario}` which is not a defined scenario, " 
                    "please choose from {{'class', 'domain', 'task-agnostic'}}.")

    return scenario, classes_names