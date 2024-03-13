import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)


@registry.register_model("mini_gpt4")
class MiniGPT4(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna0": "configs/models/minigpt4_vicuna0.yaml",
        "pretrain_llama2": "configs/models/minigpt4_llama2.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        has_qformer=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        lora_r=0,
        lora_target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        linear=False,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        self.linear = linear

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        self.has_qformer = has_qformer
        if self.has_qformer:
            print('Loading Q-Former')
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features
            )
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.load_from_pretrained(url_or_filename=q_former_model)

            if freeze_qformer:
                for name, param in self.Qformer.named_parameters():
                    param.requires_grad = False
                self.Qformer = self.Qformer.eval()
                self.Qformer.train = disabled_train
                self.query_tokens.requires_grad = False
                logging.info("freeze Qformer")

            img_f_dim = self.Qformer.config.hidden_size
            print('Loading Q-Former Done')
        else:
            img_f_dim = self.visual_encoder.num_features * 4
            print('Do not use Q-Former here.')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False, padding_side='right')
        self.llama_tokenizer.pad_token = "$$"

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        if lora_r > 0:
            self.llama_model = prepare_model_for_int8_training(self.llama_model)
            loraconfig = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llama_model = get_peft_model(self.llama_model, loraconfig)

            # if ckpt_path:
            #     print('load the llm under lora')
            #     ckpt = torch.load(ckpt_path)
            #     set_peft_model_state_dict(self.llama_model,ckpt)
            self.llama_model.print_trainable_parameters()

        else:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            img_f_dim, self.llama_model.config.hidden_size
        )
        if linear:
            self.linear_cls = torch.nn.Linear(4096, 200)
            # self.linear_cls = torch.nn.Linear(1408, 100)
            # self.order1 = ['seashore', 'scoreboard', 'plunger', 'chest', 'Persian_cat', 'candle', 'steel_arch_bridge', 'bathtub', 'fur_coat', 'gondola', 'remote_control', 'oboe', 'barrel', 'Egyptian_cat', 'beach_wagon', 'wok', 'pretzel', 'lesser_panda', 'iPod', 'koala', 'cardigan', 'punching_bag', 'albatross', 'abacus', 'snail', 'convertible', 'chimpanzee', 'mantis', 'pomegranate', 'Labrador_retriever', 'jellyfish', 'dumbbell', 'academic_gown', 'wooden_spoon', 'German_shepherd', 'space_heater', 'pill_bottle', 'kimono', 'sea_slug', 'vestment', 'fountain', 'gasmask', 'brain_coral', 'sea_cucumber', 'espresso', 'lawn_mower', 'sombrero', 'sunglasses', 'stopwatch', 'cockroach', 'sandal', 'refrigerator', 'tarantula', 'Christmas_stocking', 'banana', 'American_lobster', 'cougar', 'potpie', 'torch', 'poncho', 'beacon', 'gazelle', 'go-kart', 'black_widow', 'hog', 'sock', 'bighorn', 'monarch', 'sports_car', 'umbrella', 'altar', 'king_penguin', 'cash_machine', 'tractor', 'fly', 'bell_pepper', 'teddy', 'barbershop', 'moving_van', 'European_fire_salamander', 'birdhouse', 'guacamole', 'hourglass', 'bucket', 'orange', 'comic_book', 'bannister', 'backpack', 'dragonfly', 'crane', 'school_bus', 'brown_bear', 'snorkel', 'thatch', 'picket_fence', 'bullfrog', 'drumstick', 'golden_retriever', 'black_stork', 'goldfish', 'lemon', 'alp', 'trilobite', 'dugong', 'grasshopper', 'tabby', 'cliff', 'police_van', 'scorpion', 'pizza', 'meat_loaf', 'basketball', 'boa_constrictor', 'standard_poodle', 'mushroom', 'African_elephant', 'walking_stick', 'teapot', 'water_tower', 'spider_web', 'binoculars', 'cannon', 'bullet_train', 'lifeboat', 'guinea_pig', 'sulphur_butterfly', 'frying_pan', 'pay-phone', 'flagpole', 'acorn', 'ladybug', 'jinrikisha', 'military_uniform', 'freight_car', 'sewing_machine', 'lakeside', 'bison', 'suspension_bridge', 'beer_bottle', 'lion', 'desk', 'parking_meter', 'broom', 'rugby_ball', 'beaker', 'baboon', 'centipede', 'coral_reef', 'miniskirt', 'projectile', 'swimming_trunks', 'confectionery', 'tailed_frog', 'slug', 'dining_table', 'pop_bottle', 'mashed_potato', 'reel', 'Yorkshire_terrier', 'apron', 'cauliflower', 'Chihuahua', 'computer_keyboard', 'goose', 'spiny_lobster', 'dam', 'butcher_shop', 'pole', 'ox', 'volleyball', 'orangutan', 'triumphal_arch', 'bee', 'barn', 'water_jug', 'ice_lolly', 'turnstile', 'trolleybus', 'cliff_dwelling', 'Arabian_camel', 'bow_tie', 'CD_player', 'nail', 'American_alligator', 'lampshade', 'neck_brace', 'syringe', 'viaduct', 'ice_cream', 'rocking_chair', 'obelisk', 'chain', 'brass', 'magnetic_compass', 'maypole', 'limousine', 'bikini', 'plate', "potter's_wheel", 'organ']
            self.order1 = ['banana', 'bucket', 'goldfish', 'barn', 'pineapple', 'ant', 'grand_piano', 'husky', 'hatchet', 'mobile_phone', 'grasshopper', 'fire_engine', 'leopard', 'labrador_retriever', 'cheeseburger', 'flute', 'cannon', 'espresso', 'bagel', 'pomegranate', 'joystick', 'submarine', 'gasmask', 'revolver', 'bathtub', 'lab_coat', 'mitten', 'lorikeet', 'assault_rifle', 'castle', 'carousel', 'jeep', 'king_penguin', 'bald_eagle', 'hammer', 'ice_cream', 'vase', 'bell_pepper', 'ostrich', 'jellyfish', 'fly', 'red_fox', 'tennis_ball', 'american_egret', 'toy_poodle', 'lipstick', 'cabbage', 'sandal', 'dalmatian', 'birdhouse', 'golden_retriever', 'badger', 'yorkshire_terrier', 'ambulance', 'schooner', 'spider_web', 'hermit_crab', 'hen', 'canoe', 'wine_bottle', 'basset_hound', 'broccoli', 'fox_squirrel', 'beer_glass', 'junco', 'llama', 'rugby_ball', 'acorn', 'cockroach', 'goose', 'chow_chow', 'cauldron', 'pretzel', 'mushroom', 'basketball', 'hammerhead', 'boston_terrier', 'backpack', 'whippet', 'flamingo', 'wood_rabbit', 'cheetah', 'pembroke_welsh_corgi', 'lemon', 'volcano', 'great_white_shark', 'bloodhound', 'school_bus', 'orangutan', 'broom', 'vulture', 'tank', 'italian_greyhound', 'scuba_diver', 'baseball_player', 'standard_poodle', 'mantis', 'newt', 'sea_lion', 'parachute', 'timber_wolf', 'chimpanzee', 'cucumber', 'axolotl', 'scottish_terrier', 'candle', 'lighthouse', 'gorilla', 'killer_whale', 'ladybug', 'lawn_mower', 'cobra', 'mailbox', 'saxophone', 'hyena', 'koala', 'soccer_ball', 'polar_bear', 'black_swan', 'strawberry', 'harp', 'monarch_butterfly', 'starfish', 'dragonfly', 'border_collie', 'puffer_fish', 'pig', 'hotdog', 'rottweiler', 'bee', 'german_shepherd_dog', 'tiger', 'beaver', 'hippopotamus', 'afghan_hound', 'lion', 'goldfinch', 'lobster', 'centipede', 'peacock', 'space_shuttle', 'grey_whale', 'pelican', 'toucan', 'guillotine', 'pomeranian', 'boxer', 'bison', 'accordion', 'eel', 'tabby_cat', 'gazelle', 'collie', 'pickup_truck', 'Granny_Smith', 'west_highland_white_terrier', 'tree_frog', 'porcupine', 'clown_fish', 'snow_leopard', 'bow_tie', 'saint_bernard', 'weimaraner', 'meerkat', 'guinea_pig', 'tractor', 'military_aircraft', 'beagle', 'missile', 'chihuahua', 'binoculars', 'scorpion', 'pug', 'electric_guitar', 'shih_tzu', 'cocker_spaniels', 'violin', 'baboon', 'skunk', 'duck', 'zebra', 'gibbon', 'snail', 'iguana', 'steam_locomotive', 'stingray', 'pirate_ship', 'burrito', 'harmonica', 'wheelbarrow', 'panda', 'tarantula', 'scarf', 'cowboy_hat', 'pizza', 'African_chameleon', 'french_bulldog', 'hummingbird', 'trombone', 'shield']
            # for name, param in self.llama_proj.named_parameters():
            #     param.requires_grad = False

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image):
        # import pdb; pdb.set_trace()
        device = image.device
        # if self.low_resource:
        #     self.vit_to_cpu()
        #     image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            if self.has_qformer:
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

                inputs_llama = self.llama_proj(query_output.last_hidden_state)
            else:
                image_embeds = image_embeds[:, 1:, :]
                bs, pn, hs = image_embeds.shape
                image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))

                inputs_llama = self.llama_proj(image_embeds)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def get_context_emb(self, prompt, img_list):
        device = img_list[0].device
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def prompt_wrap(self, img_embeds, atts_img, prompts):
        if prompts:
            emb_lists = []
            if isinstance(prompts, str):
                prompts = [prompts] * len(img_embeds)

            for each_img_embed, each_prompt in zip(img_embeds, prompts):
                p_before, p_after = each_prompt.split('<ImageHere>')

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_before_embed = self.embed_tokens(p_before_tokens.input_ids)
                p_after_embed = self.embed_tokens(p_after_tokens.input_ids)
                wrapped_emb = torch.cat([p_before_embed, each_img_embed[None], p_after_embed], dim=1)
                emb_lists.append(wrapped_emb)
            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))
            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=img_embeds.device)
            for i, emb in enumerate(emb_lists):
                wrapped_embs[i, :emb_lens[i]] = emb
                wrapped_atts[i, :emb_lens[i]] = 1
            return wrapped_embs, wrapped_atts
        else:
            return img_embeds, atts_img

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        input_lens = []
        cat_embs = []
        cat_atts = []
        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )
        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens
        
    def eval_forward(self, image):
        img_embeds, atts_img = self.encode_img(image)
        # device = image.device
        # with self.maybe_autocast():
        #     img_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
        img_embeds = img_embeds.to(torch.float32)
        logits = self.linear_cls(img_embeds)
        temp=[]
        for i in range((logits.shape[0])):
            temp.append(logits[i][0].unsqueeze(0))
        logits  = torch.cat(temp, dim=0)
        return logits

    def forward(self, samples):
        if self.linear:
            image = samples["image"]
            img_embeds, atts_img = self.encode_img(image)
            # device = image.device
            # with self.maybe_autocast():
            #     img_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            logits = self.linear_cls(img_embeds)
            temp=[]
            for i in range((logits.shape[0])):
                temp.append(logits[i][0].unsqueeze(0))
            logits  = torch.cat(temp, dim=0)
            targets = torch.zeros(logits.shape[0])
            for i, t in enumerate(samples["answer"]):
                targets[i] = self.order1.index(t[21:-1])
            targets = targets.type(torch.LongTensor).to(self.device)
            # import pdb;pdb.set_trace()
            initial=20
            increment=20
            max_target = max(targets)
            if max_target-initial < 0:
                start = 0
                end = initial
            else:
                start = initial + (((max_target-initial) // increment)*increment)
                end =  initial + ((((max_target-initial) // increment)+1)*increment)
            # logits[:,0:start].fill_(float("-inf"))
            logits[:,end:].fill_(float("-inf"))
            loss = nn.functional.cross_entropy(logits, targets)
            # if loss > 100:
            #     import pdb;pdb.set_trace()
            return {"loss": loss}

        # import pdb; pdb.set_trace()
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)

        if self.prompt_list:
            instruction = random.choice(self.prompt_list)
        else:
            instruction = samples["instruction_input"] if "instruction_input" in samples else None

        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, instruction)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["answer"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(img_embeds, atts_img, to_regress_embeds, to_regress_tokens.attention_mask)
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

        part_targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        targets = (
            torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                       dtype=torch.long).to(image.device).fill_(-100)
        )

        for i, target in enumerate(part_targets):
            targets[i, input_lens[i] + 1:input_lens[i] + len(target) + 1] = target  # plus 1 for bos

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    def embed_tokens(self, token_ids):
        if hasattr(self.llama_model.base_model, 'model'): ## lora wrapped model
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        has_qformer = cfg.get("has_qformer", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 0)
        lora_alpha = cfg.get("lora_alpha", 32)
        linear = cfg.get("linear", False)

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            has_qformer=has_qformer,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            linear=linear
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
