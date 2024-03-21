# Generative Multi-modal Models are Good Class Incremental Learners

## Getting Started

### Installation

Git clone our repository, creating a python environment and activate it via the following command

```bash
git clone https://github.com/DoubleClass/GMM
cd MiniGPT-4
conda env create -n GMM python=3.9
conda install --yes --file requirements.txt
conda activate GMM
```

### Vicuna
You can get the LLM Vicuna in [huggingface](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main), or you can Dowload it via [baidunetdisk](https://pan.baidu.com/s/1jRUhVh4yv_ysItO6rQwcMw
). (code: s3pu)


Then set the downloaded vicuna folder path [here](minigpt4/configs/models/minigpt4_vicuna0.yaml) and the initial checkpoint [here](train_configs/minigpt4_stage2_finetune.yaml#L9)

### EVA_VIT_G
The code will automatically downloading the eva_vit_g.pth, we alse put it [here](https://pan.baidu.com/s/1kyc6gp7f2CXkocljhERKVg?pwd=2mux), you can manually download it and put it in 'root/.cache/torch/hub/checkpoints/eva_vit_g.pth'

### bert-base-uncased
The code will automatically downloading this, but in case you don't have access to huggingface, we also put it [here](https://pan.baidu.com/s/1XzAidcFinjsNxdz58M465w?pwd=b98f), you can manually download it and put it in '~/.cache/huggingface/hub/models--bert-base-uncased'
### datasets
#### ImageNet-R
You can download it [here](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)

Then set the dataset folder path [here](clip_base/datasets.py#L281)

Besides, you need to customize the dataset for the GPT fine-tuning process. We prepare a example here you can follow. [baidu](https://pan.baidu.com/s/1xMkqOiSylWyKY74Oef4h4g?pwd=yyea)

After downloaded the customized dataset, you can set the data root path [here](minigpt4/configs/datasets/cc_sbu/align.yaml#L7) and the indexing file [here](minigpt4/datasets/builders/image_text_pair_builder.py#L121)


## Training

After setting all model and dataset config, you can run the following command to start fine-tuning.

```bash
python train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

## Testing
After training, you will get a model checkpoint of the last continual learning stage. put the path to scipts in eval_all.sh and specify a results directory.

Run the script:

```bash 
bash eval_all.sh

```

