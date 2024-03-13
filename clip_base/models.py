from omegaconf import DictConfig

import clip
import torch
import torch.nn as nn
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
_tokenizer = _Tokenizer()
import math
from .utils import get_class_ids_per_task
from tqdm import tqdm

class ClassIncremental(nn.Module):
    def __init__(self, cfg, device, classnames, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = classnames
        self.model, self.transforms = clip.load(cfg.model_name, device=device, jit=jit)
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        # import pdb; pdb.set_trace()
        ## stuff from coop:
        self.prompt_learner = PromptLearner(cfg, device, classnames, self.model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = self.model.visual
        self.text_encoder = TextEncoder(self.model)
        self.logit_scale = self.model.logit_scale
        self.dtype = self.model.dtype
        
        # import pdb; pdb.set_trace()

        self.has_attention = cfg.has_attention
        if self.has_attention:
            # self.attention = SelfAttention(512, 512, 512, 0.01)
            self.attention = ResidualAttentionBlock(768, 12, None)
            self.ln_post = LayerNorm(768).to(self.device)
            scale = 768 ** -0.5
            self.proj = nn.Parameter(scale * torch.randn(768, 512)).to(self.device)
            self.attention = self.attention.to(self.device)
        elif cfg.linear_probe:
            # just linear probe
            self.classifer = ClipMapping(512, cfg.initial_increment).to(self.device)
        elif cfg.task_cls:
            # just linear probe
            self.classifer = ClipMapping(512, 10).to(self.device)
        # elif cfg.logits_cls:
        #     self.logits_cls = cfg.logits_cls
        #     # just linear probe
        #     self.classifer = ClipMapping(512, 10).to(self.device)
        else:
            self.clip_mapping = ClipMapping(feat_in=512, num_vector_dim=512)
            self.clip_mapping = self.clip_mapping.to(self.device)
        # self.act = nn.GELU()
        # self.another_mapping = ClipMapping(1024, 512)
        # self.another_mapping = self.another_mapping.to(self.device)

        self.base = cfg.initial_increment
        self.increment = cfg.increment
        self.linear_probe = cfg.linear_probe
        # self.logits_cls = cfg.logits_cls
        # self.ood = ood
        # self.ood_classifier = ClipMapping(feat_in=512, num_vector_dim=2)
        # self.ood_classifier = self.ood_classifier.to(self.device)
        # self.mid_ood = ClipMapping(512, 512)
        # self.mid_ood = self.mid_ood.to(self.device)

        self.all_clip = []
        self.all_logits_scale = []
        self.all_tokens = []
        self.all_classifer = []
        self.cur_cls = cfg.initial_increment
        self.current_class_names += self.classes_names[:self.cur_cls] 
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            self.text_features = self.model.encode_text(self.text_tokens).to(self.device)
        # self.text_features /= self.text_features.norm(dim=-1, keepdim=True)


        # if self.base == 50:
        #     path = '/home/ubuntu/code/ood-clip/'
        #     self.classifer.fc.weight = torch.load(path+'fc.pt')
        #     self.classifer.fc_in.weight = torch.load(path + 'fc_in.pt')
        #     self.classifer.fc_in.bias = torch.load(path + 'fc_in_bias.pt')
        # #     path = '/home/ubuntu/code/ood-clip/clip-cil/'
        # #     self.classifer.fc.weight = torch.load(path+'img-B50-fc_in-bias.pt')
        # #     self.classifer.fc_in.weight = torch.load(path + 'img-B50-fc_in-weight.pt')
        # #     self.classifer.fc_in.bias = torch.load(path + 'img-B50-fc_in-bias.pt')
        #     self.classifer = self.classifer.to(device)
        
        self.only_clip = cfg.only_clip
        self.task_cls = cfg.task_cls
        
        self.ood = cfg.ood
        if self.ood:
            self.ood_classifer = ClipMapping_OOD(512, cfg.initial_increment).to(self.device)
        self.reuse = cfg.reuse
        self.zero = cfg.zero
        self.prompt_tun = cfg.prompt_tun
        self.cur_cls = cfg.initial_increment
        self.only_cls = cfg.only_cls

        if cfg.has_another:
            self.clip_mapping = ClipMapping2fc(512, 512).to(self.device)
        self.has_another = cfg.has_another
        # import pdb; pdb.set_trace()
    
    def forward(self, image, is_train=True, task_id=0):
        if self.linear_probe:
            # linear probe, just using a fc layer to classify
            image_features = self.image_encoder(image.type(self.dtype))
            fea_for_encoded = image_features.type(torch.FloatTensor).to(self.device)
            fea_for_encoded_new = self.classifer(fea_for_encoded)
            fea_for_encoded_new = fea_for_encoded_new.type(self.dtype)
            return fea_for_encoded_new, image_features
        if self.task_cls: # or self.logits_cls:
            # ood ablation1: using a seprated classifier to classify tasks
            image_features = self.image_encoder(image.type(self.dtype))
            fea_for_encoded = image_features.type(torch.FloatTensor).to(self.device)
            fea_for_encoded_new = self.classifer(fea_for_encoded)
            fea_for_encoded_new = fea_for_encoded_new.type(self.dtype)
            return fea_for_encoded_new, image_features
        if self.zero:
            image_features = self.image_encoder(image.type(self.dtype))
            fea_for_encoded = image_features.type(self.dtype)
            fea_for_encoded /= fea_for_encoded.norm(dim=-1, keepdim=True)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * fea_for_encoded @ self.text_features.t()
            return logits, fea_for_encoded
        if self.reuse != 0:
            # if task_id > 0 and not is_train:
            #     import pdb;pdb.set_trace()
            image_features, mid_fea = self.image_encoder(image.type(self.dtype))
            fea_for_encoded = image_features.type(torch.FloatTensor).to(self.device)
            if self.has_attention:
                # import pdb; pdb.set_trace()
                mid_fea = mid_fea.type(torch.FloatTensor).to(self.device)
                mid = self.attention(mid_fea)
                mid = mid.permute(1, 0, 2)
                mid = self.ln_post(mid[:, 0, :])
                mid = mid @ self.proj

                fea_for_encoded = mid
                
            elif self.prompt_tun:
                pass
            else:
                fea_for_encoded = self.clip_mapping(fea_for_encoded)

            fea_for_encoded = fea_for_encoded.type(self.dtype)
            image_features = fea_for_encoded

            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            return logits, image_features
        if self.ood and is_train:
            # import pdb; pdb.set_trace()
            # if task_id > 0:
            #     import pdb; pdb.set_trace()
            image_features, in_fea = self.image_encoder(image.type(self.dtype))
            fea_for_encoded = image_features.type(torch.FloatTensor).to(self.device)
            fea_for_encoded_new = self.ood_classifer(fea_for_encoded)
            fea_for_encoded_new = fea_for_encoded_new.type(self.dtype)
            return fea_for_encoded_new, image_features
            
        else:
            # testing code ood            
            # import pdb; pdb.set_trace()
            if self.ood_classifer.fc.weight.shape[0] == self.base and task_id==0:
                # first task
                image_features, in_fea = self.image_encoder(image.type(self.dtype))
                fea_for_encoded = image_features.type(torch.FloatTensor).to(self.device)
                fea_for_encoded_new = self.ood_classifer(fea_for_encoded)
                return fea_for_encoded_new, image_features
        
            elif self.only_clip and self.ood_classifer.fc.weight.shape[0] == self.base:
                # only clip as the ood detector
                old_classifier = deepcopy(self.ood_classifer)
                text_tokens = clip.tokenize(
                    [self.prompt_template.format(c) for c in self.current_class_names]
                ).to(self.device)
                # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.current_class_names]).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                image_features, _ = self.image_encoder(image.type(self.dtype))
                results = []
                for idx_im in range(len(image_features)):
                    each_feature = image_features[idx_im] / image_features[idx_im].norm(dim=-1, keepdim=True)
                    each_fea_for_encoded = each_feature.type(torch.FloatTensor).to(self.device)
                    prompts = self.prompt_learner()

                    idx_count = 0
                    i = len(self.all_classifer) - 1
                    with torch.no_grad():
                        similarity = (100.0 * each_feature @ text_features.T).softmax(dim=-1)
                    values, indices = similarity.topk(2)
                    # import pdb; pdb.set_trace()
                    ## temp code
                    cur_base = int(indices[0]//10 * 10)
                    cur_end = cur_base + 10
                    cur_cls_idx = cur_base // 10
                    logits = self.all_classifer[cur_cls_idx](each_fea_for_encoded)
                    v, pred_tar = logits.max(dim=0)
                    results.append(int(pred_tar+cur_base))
                    ## temp code end
                # import pdb; pdb.set_trace()
                return results, None

            else:
                # second task to last task
                old_classifier = deepcopy(self.ood_classifer)
                text_tokens = clip.tokenize(
                    [self.prompt_template.format(c) for c in self.current_class_names]
                ).to(self.device)
                # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.current_class_names]).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                image_features, _ = self.image_encoder(image.type(self.dtype))
                results = []
                for idx_im in range(len(image_features)):
                    each_feature = image_features[idx_im] / image_features[idx_im].norm(dim=-1, keepdim=True)
                    each_fea_for_encoded = each_feature.type(torch.FloatTensor).to(self.device)
                    prompts = self.prompt_learner()

                    idx_count = 0
                    i = len(self.all_classifer) - 1
                    old_i = i
                    similarity = (100.0 * each_feature @ text_features.T).softmax(dim=-1)
                    while i >= 0:
                        self.ood_classifer = self.all_classifer[i]
                        logits = self.ood_classifer(each_fea_for_encoded)
                        max_value, max_idx = torch.max(logits, 0)
                        cur_tar = int(max_idx)
                        if cur_tar != self.increment and i != 0:
                            cur_base = self.base + self.increment * (i -1)
                            inside_class_names = self.classes_names[: cur_base+self.increment]
                            # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in inside_class_names]).to(self.device)
                            
                            text_tokens = clip.tokenize(
                                [self.prompt_template.format(c) for c in inside_class_names]
                            ).to(self.device)
                            
                            with torch.no_grad():
                                text_features = self.model.encode_text(text_tokens)
                            text_features /= text_features.norm(dim=-1, keepdim=True)
                            with torch.no_grad():
                                similarity = (100.0 * each_feature @ text_features.T).softmax(dim=-1)
                            values, indices = similarity.topk(2)
            
                            idx_count += 1
                            cur_range = range(cur_base, cur_base + self.increment)

                            if self.only_cls:
                                results.append(cur_tar + cur_base)
                                break
                            
                            if indices[0] in cur_range or indices[1] in cur_range: #or indices[2] in cur_range or indices[3] in cur_range:
                                results.append(cur_tar + cur_base)
                                break
                            elif i==0:                                
                                break
                            i -= 1
                        elif i == 0 and old_i != 0:

                            similarity = (100.0 * each_feature @ text_features.T).softmax(dim=-1)
                            values, indices = similarity.topk(2)
                            # import pdb; pdb.set_trace()
                            results.append(indices[0])
                            break
                        else:
                            i -= 1 
                self.ood_classifer = old_classifier
                return results, None

    def adaptation(self, task_id, increment):
        # import pdb; pdb.set_trace()
        # self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        


        self.cur_cls += increment
        if self.linear_probe:
            self.classifer = ClipMapping(512, self.cur_cls).to(self.device)
        elif self.has_attention:
            self.attention = ResidualAttentionBlock(768, 12, None)
            self.ln_post = LayerNorm(768).to(self.device)
            scale = 768 ** -0.5
            self.proj = nn.Parameter(scale * torch.randn(768, 512)).to(self.device)
            self.attention = self.attention.to(self.device)
        elif self.task_cls:
            self.classifer = ClipMapping(512, task_id + 2).to(self.device)
        # elif self.logits_cls:
        #     self.classifer = ClipMapping(512, self.cur_cls).to(self.device)
        elif self.ood:
            if self.only_clip:
                self.ood_classifer = ClipMapping_OOD(512, increment).to(self.device)
            else:
                self.ood_classifer = ClipMapping_OOD(512, increment+1).to(self.device)
        self.current_class_names = self.classes_names[:self.cur_cls]

        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(self.text_tokens).to(self.device)
        # import pdb; pdb.set_trace()
        self.text_features = self.text_features.to(self.device)
        self.prompt_learner.adapt(increment, self.model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
    
    def get_fea_for_mem(self, loader, task_id):
        all_fea = []
        if task_id == 0:
            num_of_img = self.base * 500
        else:
            num_of_img = self.base * 500 + task_id*self.increment*500

        num_of_img = len(loader.dataset)
        all_fea = torch.empty((num_of_img, 512))
        idx = 0
        with torch.no_grad():
            for x, y, t in tqdm(loader):
                _, fea = self.forward(x.to(self.device))
                try:
                    all_fea[idx:idx+x.shape[0]] = fea
                except:
                    import pdb; pdb.set_trace()
                idx += fea.shape[0]
        # all_fea = torch.zeros(num_of_img, 512)
        # import pdb; pdb.set_trace()
        return all_fea


    def get_old_model(self):
        

        if self.has_attention:
            self.old_attn = deepcopy(self.attention)
        # elif self.another_layer:
        #     self.old_another_clip = deepcopy(self.another_mapping)
        else:
            self.old_clip = deepcopy(self.clip_mapping)
            self.old_clip.eval()
        # self.old_other = deepcopy(self.another_mapping)
        # self.old_other.eval()

        self.old_tokenized_prompts = self.tokenized_prompts
        self.old_logit = self.logit_scale
        self.old_prompts = self.prompt_learner()
        self.old_prompt_learner = deepcopy(self.prompt_learner)
        self.old_prompt_learner.eval()

        if self.has_attention:
            for param in self.attention.parameters():
                param.requires_grad = False
        else:
            for param in self.old_clip.parameters():
                param.requires_grad = False
    
    def post_train(self, task_id, increment):
        # old_tokenized_prompts = deepcopy(self.tokenized_prompts)
        # self.all_tokens.append(old_tokenized_prompts)

        # cur_weight = deepcopy(self.clip_mapping.fc.weight)
        # self.all_clip.append(cur_weight)

        # cur_logits = self.logit_scale
        # self.all_logits_scale.append(deepcopy(cur_logits))
        # if self.base == 50:
        #     path = '/home/ubuntu/code/ood-clip/clip-cil/'
        #     torch.save(self.classifer.fc.weight, path +'img-B50-fc-weight.pt')
        #     torch.save(self.classifer.fc_in.weight, path +'img-B50-fc_in-weight.pt')
        #     torch.save(self.classifer.fc_in.bias, path +'img-B50-fc_in-bias.pt')
        if self.ood:
            cur_classifer = deepcopy(self.ood_classifer)
            self.all_classifer.append(cur_classifer)
            # import pdb; pdb.set_trace()
            self.current_class_names = self.classes_names[:self.base + task_id*increment]
    
    def get_old_outputs(self, image, task_id):
        image_features, _ = self.image_encoder(image.type(self.dtype))
        fea_for_encoded = image_features.type(torch.FloatTensor).to(self.device)


        if self.has_attention:
            fea_for_encoded = self.old_attn(fea_for_encoded)
        # elif self.res:
        #     fea_for_encoded_clip = self.old_clip(fea_for_encoded)

        #     # plus a residual
        #     fea_for_encoded = fea_for_encoded + fea_for_encoded_clip
            # fea_for_encoded = self.act(fea_for_encoded)
            # fea_for_encoded = self.old_other(fea_for_encoded)
        else:
            fea_for_encoded = self.old_clip(fea_for_encoded)

        fea_for_encoded = fea_for_encoded.type(self.dtype)
        # prompts = self.old_prompt_learner()
        # import pdb; pdb.set_trace()
        #prompts = deepcopy(self.prompt_for_save)

        num_cls = (task_id) *10
        # prompts = prompts[:num_cls, :,:]
        tokenized_prompts = self.old_tokenized_prompts.to(self.device)
        try:
            text_features = self.text_encoder(self.old_prompts, tokenized_prompts)
        except:
            import pdb; pdb.set_trace()
        raw_image_fea = image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.old_logit.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, raw_image_fea

class DomainIncremental(nn.Module):
    pass


class TaskAgnostic(nn.Module):
    pass



def load_model(cfg: DictConfig, device: torch.device, class_name, has_attention, res, another_layer, linear, zero) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.
    
    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.
        
    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        return ClassIncremental(cfg, device, class_name, has_attention, res, another_layer, linear, zero)
    elif cfg.scenario == "domain":
        return DomainIncremental(cfg, device)
    elif cfg.scenario == "task-aganostic":
        return TaskAgnostic(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario, 
            Please choose from ['class', "domain', 'task-agnostic']
        """)
    


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        with torch.no_grad():
            x = prompts + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, device, classnames, clip_model):
        super().__init__()
        self.device = device
        self.current_cls = cfg.initial_increment
        n_cls = cfg.initial_increment
        n_ctx = cfg.N_CTX
        ctx_init = cfg.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT_SIZE
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            # import pdb; pdb.set_trace()
            if cfg.TRAINER_COOP_CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                torch.manual_seed(2)
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                # import pdb; pdb.set_trace()
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        # import pdb; pdb.set_trace()
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.all_class_names = classnames

        classnames = classnames[:n_cls]
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.to(device)

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER_COOP_CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        ctx = ctx.to(self.device)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

    def adapt(self, increase, clip_model):
        self.current_cls += increase
        dtype = clip_model.dtype
        n_ctx = self.n_ctx
        prompt_prefix = " ".join(["X"] * n_ctx)
        classnames = self.all_class_names[self.current_cls-increase:self.current_cls]
        # import pdb; pdb.set_trace()
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", torch.cat((self.token_prefix, embedding[:, :1, :]), 0))  # SOS
        self.register_buffer("token_suffix", torch.cat((self.token_suffix, embedding[:, 1 + n_ctx :, :]), 0))  # CLS, EOS
        self.tokenized_prompts = torch.cat((self.tokenized_prompts, tokenized_prompts), 0)
        self.name_lens = [len(_tokenizer.encode(name)) for name in self.all_class_names[:self.current_cls]]
        self.n_cls += increase

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        
        logits = logit_scale * image_features @ text_features.t()

        return logits

import torch.nn.init as init

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class ClipMapping(nn.Module):
    # the projection head
    # mapping the output feature to the dimension of the anchor vector
    def __init__(self, feat_in, num_vector_dim):
        super(ClipMapping, self).__init__()
        self.fc = nn.Linear(feat_in, num_vector_dim)
        self.apply(_weights_init)

    def forward(self, x):
        x = self.fc(x)
        return x

class ClipMapping2fc(nn.Module):
    # the projection head
    # mapping the output feature to the dimension of the anchor vector
    def __init__(self, feat_in, num_vector_dim):
        super(ClipMapping2fc, self).__init__()
        self.fc1 = nn.Linear(feat_in, feat_in)
        self.fc = nn.Linear(feat_in, num_vector_dim)
        self.apply(_weights_init)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc(x)
        return x

    
class ClipMapping_OOD(nn.Module):
    # the projection head
    # mapping the output feature to the dimension of the anchor vector
    def __init__(self, feat_in, num_vector_dim):
        super(ClipMapping_OOD, self).__init__()
        self.fc_in = nn.Linear(feat_in, feat_in)
        self.fc = nn.Linear(feat_in, num_vector_dim, bias=False)
        self.apply(_weights_init)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.fc(x)
        return x

    
class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(hidden_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        # import pdb; pdb.set_trace()
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor):
        # import pdb; pdb.set_trace()
        input_tensor = input_tensor.type(torch.float32)
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

from collections import OrderedDict
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention_in(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        #import pdb; pdb.set_trace()
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        #import pdb; pdb.set_trace()
        x = x + self.attention_in(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x