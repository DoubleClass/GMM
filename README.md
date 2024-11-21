# Generative Multi-modal Models are Good Class Incremental Learners
This is the official code for our CVPR paper: <a href='https://arxiv.org/abs/2403.18383.pdf'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
## Getting Started

### Installation

Git clone our repository, creating a python environment and activate it via the following command

```bash
git clone https://github.com/DoubleClass/GMM
cd GMM
conda env create -f env_GMM.yaml
conda activate GMM
pip install git+https://github.com/openai/CLIP.git
```

### Vicuna
You can get the LLM Vicuna in [huggingface-vicuna-7b](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main), or you can Dowload it via [baidunetdisk](https://pan.baidu.com/s/1jRUhVh4yv_ysItO6rQwcMw
). (code: s3pu) or [google-drive](https://drive.google.com/drive/folders/115uMyEDgCQCvWB0K8s-FWC8_toIK3V9m?usp=sharing)


Then set the downloaded vicuna folder path [here](minigpt4/configs/models/minigpt4_vicuna0.yaml) and the initial checkpoint [here](train_configs/minigpt4_stage2_finetune.yaml#L9)

### EVA_VIT_G
The code will automatically downloading the eva_vit_g.pth, we alse put it [here](https://pan.baidu.com/s/1kyc6gp7f2CXkocljhERKVg?pwd=2mux) or [huggingface](https://huggingface.co/lainxx/eva_vit_g/blob/main/eva_vit_g.pth), you can manually download it and put it in the cache dir: `.cache/torch/hub/checkpoints`

### bert-base-uncased
The code will automatically downloading this, but in case you don't have access to [huggingface](https://huggingface.co/google-bert/bert-base-uncased/tree/main), we also put it [here](https://pan.baidu.com/s/1XzAidcFinjsNxdz58M465w?pwd=b98f) , you can manually download it and alse put it in cache dir: `.cache/huggingface/hub/models--bert-base-uncased`
### datasets
#### ImageNet-R
You can download it [here](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)

Then set the dataset folder path [here](clip_base/datasets.py#L134)

Besides, you need to customize the dataset for the GPT fine-tuning process. We prepare a example here you can follow: [download](https://pan.baidu.com/s/1xMkqOiSylWyKY74Oef4h4g?pwd=yyea) or [google-dirve](https://drive.google.com/file/d/1BbGvx8Fl1F3FKoNInOFZnhqVrP1M9u0O/view?usp=drive_link)

After downloaded the customized dataset, you can set the data root path [here](minigpt4/configs/datasets/cc_sbu/align.yaml#L7).


## Training

After setting all model and dataset config, you can run the following command to start fine-tuning.

```bash
python train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

## Testing
After training, you will get a model checkpoint of the last continual learning stage. put the path to scipts in eval_all.sh and specify a results directory.

Then set the results path in the [get_score_all.py](https://vscode.dev/github/DoubleClass/GMM/get_score_all.py#L1)

Run the script:

```bash 
bash eval_all.sh

```


## Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@article{cao2024GMM,
  title={Generative Multi-modal Models are Good Class Incremental Learners},
  author={Cao, Xusheng and Lu, Haori and Huang, Linlan and Liu, Xialei and Cheng, Ming-Ming},
  journal={IEEE Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

## License
This code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
Please note that any commercial use of this code requires formal permission prior to use.

## Contact

For technical questions, please contact <a href="caoxusheng@mail.nankai.edu.cn">caoxusheng@mail.nankai.edu.cn</a> 
