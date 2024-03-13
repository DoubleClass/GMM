import os
import json
import yaml
from omegaconf import DictConfig, OmegaConf


def get_class_order(file_name: str) -> list:
    r"""TO BE DOCUMENTED"""
    with open(file_name, "r+") as f:
        data = yaml.safe_load(f)
        return data["class_order"]

def get_ordered_class_name(class_order, class_name):
    new_class_name = []
    for i in range(len(class_name)):
        new_class_name.append(class_name[class_order[i]])
    return new_class_name

def get_class_ids_per_task(args):
    yield args.class_order[:args.initial_increment]
    for i in range(args.initial_increment, len(args.class_order), args.increment):
        yield args.class_order[i:i + args.increment]

def get_dataset_class_names( long=False):
    with open("./imagenet100_classes.txt", "r") as f:
        lines = f.read().splitlines()
    return [line.split("\t")[-1] for line in lines]


def save_config(config: DictConfig) -> None:
    OmegaConf.save(config, "config.yaml")


def get_workdir(path):
    split_path = path.split("/")
    workdir_idx = split_path.index("clip_based")
    return "/".join(split_path[:workdir_idx+1])