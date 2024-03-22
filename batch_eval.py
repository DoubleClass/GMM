import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from clip_base.datasets import build_cl_scenarios
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--task-id", type=int, default=0, help="which task you running")
    parser.add_argument("--ckpt-path", type=str, default='bad_path', help="specify the path of ckpt for this task.")
    parser.add_argument("--txt-path", type=str, default='bad_path', help="specify the path of resulst of this task.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================



conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)
setup_seeds(cfg)


model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_config.ckpt = args.ckpt_path
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), task_id=args.task_id)
print('Initialization Finished')




# prepare datasets
def get_ordered_class_name(class_order, class_name):
    new_class_name = []
    for i in range(len(class_name)):
        new_class_name.append(class_name[class_order[i]])
    return new_class_name

cfg_o = cfg.get_o_config()
_, transforms = clip.load("ViT-B/16", device='cuda:{}'.format(args.gpu_id))
eval_dataset, classes_names = build_cl_scenarios(
        cfg_o, is_train=False, transforms=transforms
    )

new_class_name = get_ordered_class_name(cfg_o.class_order, classes_names)

with open(args.txt_path, 'w') as f:
    eval_loader = DataLoader(eval_dataset[:args.task_id+1], batch_size=cfg_o.batch)
    names = new_class_name[:cfg_o.initial_increment + args.task_id * cfg_o.increment]
    for inputs, targets, task_ids in tqdm(eval_loader):

        chat_state = CONV_VISION.copy()
        img_list = []
        llm_message = chat.upload_img(inputs, chat_state, img_list)
        chat.ask('what is this photo of?', chat_state)
        llm_message = chat.answer(conv=chat_state,img_list=img_list,num_beams=1,temperature=0.01,max_new_tokens=300,max_length=2000)[0]
        for i in range(inputs.shape[0]):
            str1 = 'the label is '  + new_class_name[targets[i]] + '\n'
            str2 = 'msg: '  + llm_message[i] + '\n'
            f.write(str1)
            f.write(str2)