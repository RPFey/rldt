# Reinforcement Learning for Flow-Matching Policies with Density Transport (RLDT)

[[Paper](https://arxiv.org/abs/2606.08602)]&nbsp;&nbsp;[[Website](https://rpfey.github.io/rldt-web/)]

[Boshu Lei](https://rpfey.github.io/)<sup>1</sup>, [Kostas Daniilidis](https://www.cis.upenn.edu/~kostas/)<sup>1</sup>, [Antonio Loqercio](https://antonilo.github.io/)<sup>1</sup>

<sup>1</sup>University of Pennsylvania

<img src="https://rpfey.github.io/rldt-web/static/images/pipeline.png" alt="drawing" width="100%"/>

This repository is built on top of the [DPPO](https://github.com/irom-princeton/dppo) repository.

## Installation

1. Clone the repository
```bash
git clone https://github.com/RPFey/rldt.git
cd rldt
```

2. Install core dependencies with a conda environment (if you do not plan to use Furniture-Bench, a higher Python version such as 3.10 can be installed instead) on a Linux machine with a Nvidia GPU.
```bash
conda create -n rldt python=3.8 -y
conda activate rldt
pip install -e .
```

3. Install specific environment dependencies (Gym / Kitchen / Robomimic / D3IL / Furniture-Bench) or all dependencies (except for Kitchen, which has dependency conflicts with other tasks).
```console
pip install -e .[gym] # or [kitchen], [robomimic], [d3il], [furniture]
pip install -e .[all] # except for Kitchen
```

4. [Install MuJoCo for Gym and/or Robomimic](installation/install_mujoco.md). [Install D3IL](installation/install_d3il.md). [Install IsaacGym and Furniture-Bench](installation/install_furniture.md)

5. Set environment variables for data and logging directory (default is `data/` and `log/`), and set WandB entity (username or team name)
```
source script/set_path.sh
```

## Usage - Pre-training

**Note**: You may skip pre-training if you would like to use the default checkpoint (available for download) for fine-tuning.
For pre-training data, they can be found at [here](https://drive.google.com/drive/folders/1AXZvNQEKOrp0_jk1VLepKh_oHCg_9e3r?usp=drive_link).

Pre-training script will download the data (including normalization statistics) automatically to the data directory.
<!-- The data path follows `${DPPO_DATA_DIR}/<benchmark>/<task>/train.npz`, e.g., `${DPPO_DATA_DIR}/gym/hopper-medium-v2/train.npz`. -->

### Run pre-training with data
All the configs can be found under `cfg/<env>/pretrain/`. A new WandB project may be created based on `wandb.project` in the config file; set `wandb=null` in the command line to test without WandB logging.
<!-- To run pre-training, first set your WandB entity (username or team name) and the parent directory for logging as environment variables. -->

<!-- ```console
export DPPO_WANDB_ENTITY=<your_wandb_entity>
export DPPO_LOG_DIR=<your_prefered_logging_directory>
``` -->
```console
# Gym - hopper/walker2d/halfcheetah
script/run.py --config-name=pre_flowmatching_mlp \
            --config-dir=cfg/gym/pretrain/walker2d-medium-v2

# Robomimic - square/transport
python script/run.py --config-name=pre_flowmatching_mlp_img \
    --config-dir=cfg/robomimic/pretrain/square

# Furniture-Bench - one_leg/lamp/round_table_low/med
python script/run.py --config-name=pre_flowmatching_mlp \
    --config-dir=cfg/furniture/pretrain/one_leg_low
```

See [here](cfg/pretraining.md) for details of the experiments in the paper.

## Usage - Fine-tuning

<!-- ### Set up pre-trained policy -->

<!-- If you did not set the environment variables for pre-training, we need to set them here for fine-tuning. 
```console
export DPPO_WANDB_ENTITY=<your_wandb_entity>
export DPPO_LOG_DIR=<your_prefered_logging_directory>
``` -->
<!-- First create a directory as the parent directory of the downloaded checkpoints and set the environment variable for it.
```console
export DPPO_LOG_DIR=/path/to/checkpoint
``` -->

Pre-trained policies used in the paper can be found [here](https://drive.google.com/drive/folders/1ZlFqmhxC4S8Xh1pzZ-fXYzS5-P8sfpiP?usp=drive_link). Fine-tuning script will download the default checkpoint automatically to the logging directory.
<!-- For RLDT pre-trained checkpoints, they can be found at [here](). -->
<!-- or you may manually download other ones (different epochs) or use your own pre-trained policy if you like. -->

<!-- e.g., `${DPPO_LOG_DIR}/gym-pretrain/hopper-medium-v2_pre_diffusion_mlp_ta4_td20/2024-08-26_22-31-03_42/checkpoint/state_0.pt`. -->

<!-- The checkpoint path follows `${DPPO_LOG_DIR}/<benchmark>/<task>/.../<run>/checkpoint/state_<epoch>.pt`. -->

### Fine-tuning pre-trained policy

All the configs can be found under `cfg/<env>/finetune/`. A new WandB project may be created based on `wandb.project` in the config file; set `wandb=null` in the command line to test without WandB logging.
<!-- Running them will download the default pre-trained policy. -->
<!-- Running the script will download the default pre-trained policy checkpoint specified in the config (`base_policy_path`) automatically, as well as the normalization statistics, to `DPPO_LOG_DIR`.  -->
```console
# Gym - hopper/walker2d/halfcheetah
python script/run.py --config-name=ft_svgd_flowmatching_mlp \
    --config-dir=cfg/gym/finetune/hopper-v2

# Robomimic - square/transport
python script/run.py --config-name=ft_svgd_flowmatching_mlp_img_vit \
    --config-dir=cfg/robomimic/finetune/can

# Furniture-Bench - one_leg/lamp/round_table_low/med
python script/run.py --config-name=ft_svgd_fm_mlp_tcond \
    --config-dir=cfg/furniture/finetune/one_leg_low
```

**Note**: For the Robomimic task, we have a parameter `mini_batch_split` . Please increase this parameter if you have GPU OOM issue. We test our configuration on A40 and L40 GPUs.

<!-- **Note**: If you did not download the pre-training [data](https://drive.google.com/drive/folders/1AXZvNQEKOrp0_jk1VLepKh_oHCg_9e3r?usp=drive_link), you need to download the normalization statistics from it for fine-tuning, e.g., `${DPPO_DATA_DIR}/furniture/round_table_low/normalization.pkl`. -->

<!-- See [here](cfg/finetuning.md) for details of the experiments in the paper. -->

### Visualization
* Furniture-Bench tasks can be visualized in GUI by specifying `env.specific.headless=False` and `env.n_envs=1` in fine-tuning configs.
* D3IL environment can be visualized in GUI by `+env.render=True`, `env.n_envs=1`, and `train.render.num=1`. There is a basic script at `script/test_d3il_render.py`.
* Videos of trials in Robomimic tasks can be recorded by specifying `env.save_video=True`, `train.render.freq=<iterations>`, and `train.render.num=<num_video>` in fine-tuning configs.

## License
This repository is released under the MIT license. See [LICENSE](LICENSE).

## Acknowledgement
This repository is built largely upon the [DPPO](https://github.com/irom-princeton/dppo) repository with following:
* [Diffuser, Janner et al.](https://github.com/jannerm/diffuser): general code base and DDPM implementation
* [Diffusion Policy, Chi et al.](https://github.com/real-stanford/diffusion_policy): general code base especially the env wrappers
* [CleanRL, Huang et al.](https://github.com/vwxyzjn/cleanrl): PPO implementation
* [IBRL, Hu et al.](https://github.com/hengyuan-hu/ibrl): ViT implementation
* [D3IL, Jia et al.](https://github.com/ALRhub/d3il): D3IL benchmark
* [Robomimic, Mandlekar et al.](https://github.com/ARISE-Initiative/robomimic): Robomimic benchmark
* [Furniture-Bench, Heo et al.](https://github.com/clvrai/furniture-bench): Furniture-Bench benchmark
* [AWR, Peng et al.](https://github.com/xbpeng/awr): DAWR baseline (modified from AWR)
* [DIPO, Yang et al.](https://github.com/BellmanTimeHut/DIPO): DIPO baseline
* [IDQL, Hansen-Estruch et al.](https://github.com/philippe-eecs/IDQL): IDQL baseline
* [DQL, Wang et al.](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL): DQL baseline
* [QSM, Psenka et al.](https://www.michaelpsenka.io/qsm/): QSM baseline
* [Score SDE, Song et al.](https://github.com/yang-song/score_sde_pytorch/): diffusion exact likelihood
* [DPPO, Allen et al.](https://github.com/irom-princeton/dppo): DPPO