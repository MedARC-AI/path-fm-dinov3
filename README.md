# Path-FM-DINOv3

[![Collaborate with us on Discord](https://img.shields.io/badge/Discord-Collaborate%20with%20us-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/tVR4TWnRM9)

In-progress development expanding upon our [OpenMidnight](https://github.com/MedARC-AI/OpenMidnight) computational pathology foundation model where we are migrating from DINOv2 to DINOv3, implementing better whole-slide context, improving dataset preprocessing, among other things.

This is a publicly developed, open-source project by [MedARC](https://www.medarc.ai/). If you are interested in helping out, [join our Discord server](https://discord.gg/tVR4TWnRM9) and introduce yourself in our `#path-fm` channel.

## Contributors

<!--
Add yourself when you make a PR in alphabetical order.
If your info isn't right, please open a PR to fix it.
Format: [First name you like to be referred to as | @your_discord_ID_on_MedARC](whatever link you like) `
-->
A (growing) list of our past and current contributors:
- [Benjamin | @benjamin_w](https://github.com/warner-benjamin),  
- [Connor | @connortslane](https://github.com/clane9),
- [Daniel | @KublaiKhan1](https://github.com/daniel-z-kaplan),
- [Dojo | @Dojo2024](https://github.com/Dojo2024),
- [Leema | @Leema-Krishna-Murali](https://github.com/Leema-Krishna-Murali),
- [Navneel | @NavneelSinghal](https://github.com/NavneelSinghal),
- [Paul | @paulscotti](https://github.com/PaulScotti),
- [Ratna | @sagarigrandhi](https://github.com/sagarigrandhi),
- [Tanishq | @ilovescience](https://github.com/stmabraham),
- [Zonun | @zonunmawia7](https://github.com/Nuna7),

Thank you! 🤗

# Installation

Clone the repository:

```bash
git clone https://github.com/MedARC-AI/path-fm-dinov3.git
```

Change into the directory, then run the installation script:

```bash
./install.sh
```

This will create a virtual environment, "dinov3", with all necessary packages pre-installed, located as a .venv folder in the same directory as path-fm-dinov3. It will automatically detect your machine's CUDA version and install packages appropriately.

This install.sh script also downloads [our modified version of Meta's ViT-H+/16 checkpoint](https://huggingface.co/datasets/medarc/path-fm-dinov3/blob/main/dinov3_vith16plus_saved_teacher.pth) and places it into your `path-fm-dinov3/checkpoints/` folder. 

Note: We have only personally verified training works as intended with this repository using H100 GPUs.

Then, activate your environment and setup your wandb:

```bash
source .venv/bin/activate
wandb init
```

By default, we log model training to wandb. Run `wandb init` inside of `path-fm-dinov3/` before starting your training run so that wandb is properly configured.

You can now run one of our `run*.sh` scripts to train your model (see Training section below), using the YAML config specified in that script.

Once you have successfully completed model training and have a teacher_checkpoint.pth, you can evaluate it using [Kaiko.AI's eva framework](https://github.com/kaiko-ai/eva) and the [Mahmood Lab's HEST benchmark](https://github.com/mahmoodlab/HEST) (skip to the Evaluation section below).

# Training

We use `run*.sh` to initiate training runs via torchrun. Depending on your compute setting, modify and run the corresponding shell script described in the relevant subsection below.

We use YAML configs for specifying important hyperparameters during training. These files are located in `dinov3/configs/train/` (not in `eval_configs/`, which stores YAML configs for eva evaluation benchmarking). Our replication checkpoint specifically used `dinov3/configs/train/vitg14_reg4.yaml`.

There are some variables that are specified in `run*.sh` directly (as opposed to the YAML config), such as the output directory for saving checkpoints and the specific CUDA devices you want to train with.

If you are getting rate limited by huggingface, one easy method to increase your rate is to first `export HF_TOKEN=<your HF token here>` before running your code (https://huggingface.co/settings/tokens).

## Dataset prep

The easiest way to get started with model training is to stream data from [our huggingface dataset](https://huggingface.co/datasets/medarc/TCGA-12K-parquet-shuffled). Enable/disable streaming from huggingface via your YAML config (`train.streaming_from_hf=[True,False]`). 

If you want to use our TCGA-12K-parquet-shuffled dataset locally rather than streaming, keep `train.streaming_from_hf=True` but set `train.streaming_dataset.path=<full_path_to_local_TCGA-12K-parquet-shuffled>`.

If you disable `train.streaming_from_hf`, you will need to first locally download the TCGA-12K dataset (~12TB) and then use the scripts provided in `prepatching_scripts/` to create a txt file containing the svs filepaths and locations/magnitude from which to create patches on-the-fly during model training. You can alternatively use [our original sample_dataset_30.txt file](https://huggingface.co/SophontAI/Path-fm-dinov3/blob/main/sample_dataset_30.txt), but note you would need to modify that txt to correct its use of absolute filepaths.

## Training Single GPU (Short Run)

```bash
./run_1gpu.sh
```

We are working on a YAML config to support an informative, short training run on a single GPU that can be completed in under 12 hours. We hope this can be particularly useful for debugging and ablation experiments.

The current `run_1gpu.sh` uses `dinov3/configs/train/vith16plus_1gpu.yaml`, which is the same as the single node 8-GPU YAML but with batch size lowered to 44, epochs lowered to 20, warmup_epochs lowered to 5, no early stopping, and FSDPCheckpointing disabled (so resuming an interrupted run will not work). It takes around 2 hours to run on 1 H100.

## Training Single Node, Multi‑GPU (Full Run)

```bash
./run_8gpus.sh
```

Train across multiple GPUs on a single node.

# Downstream Evaluation

## eva Benchmarks

Ensure you have a `teacher_checkpoint.pth` checkpoint ready to be evaluated before running benchmarks.

Then, `cd` into the same `path-fm-dinov3` folder cloned from our Installation steps and clone our modified GitHub repo forked from the original [kaiko-eva](https://github.com/kaiko-ai/eva):

```bash
cd path-fm-dinov3
source .venv/bin/activate
git clone https://github.com/MedARC-AI/eva-probe -b dinov3
```

Then, install the eva framework to enable use of the `eva` command in your terminal (using `--no-deps` because the `path-fm-dinov3` virtual environment already contains the necessary packages):

```bash
uv pip install -e './eva-probe[vision]' --no-deps
```

We provide a YAML config file for every eva dataset in `path-fm-dinov3/eval_configs/`. Not every dataset permits automatic downloading. For datasets like BACH that do, you can automatically download the dataset by specifying `DOWNLOAD_DATA=true` when calling `eva predict_fit`. For other datasets, follow the manual download steps described in the [eva datasets documentation](https://kaiko-ai.github.io/eva/main/datasets/). Consult each dataset's specific YAML config for details on whether automatic downloading is supported; if not, modify the config to provide the path to your dataset prior to eva benchmarking.

Below are the steps for running the [BACH](https://kaiko-ai.github.io/eva/main/datasets/bach/) evaluation. 

```bash
cd eva-probe # should be located in path-fm-dinov3/eva-probe
CUDA_VISIBLE_DEVICES=0 CHECKPOINT_PATH=<path_to_your_teacher_checkpoint> DATA_ROOT=<path_to_bach_dataset_folder> eva predict_fit --config ../eval_configs/bach.yaml 
```

All eva evaluations should be run on a single GPU by setting `CUDA_VISIBLE_DEVICES=0`. We observed inconsistent and worse results when trying to evaluate using multiple GPUs.

## HEST Benchmark

First ensure you have a checkpoint ready to be evaluated. Place the .pth file for your teacher model in the /checkpoints folder. You can download our pretrained checkpoint here: *insert URL here*

Then, `cd` into the same `path-fm-dinov3` folder cloned from our Installation steps and clone the [Mahmood Lab's HEST GitHub repo](https://github.com/mahmoodlab/HEST):

```bash
cd path-fm-dinov3
source .venv/bin/activate
git clone https://github.com/mahmoodlab/HEST.git
cd HEST
git checkout afd42c3143092c51e6bcc0f1df65bbf58a467e5e
cd .. # cd back to path-fm-dinov3/ for subsequent install steps
```

Then install HEST framework so that we can import invoke their benchmark function  (using `--no-deps` because the `path-fm-dinov3` virtual environment already contains the necessary packages).

```bash
uv pip install -e ./HEST --no-deps
```

Now uncomment out specific lines in the HEST YAML config to enable PCA dimensionality reduction, benchmark across all HEST datasets used in the original paper, and solely benchmark our Dinov3 model as the encoder:

```bash
sed -i -E '/^datasets:/,/^\]/{s/^([[:space:]]*)#([[:space:]]*)"([^"]+)",/\1"\3",/;s/^([[:space:]]*)"HCC"(,?)/\1# "HCC"\2/}; s/^[[:space:]]*#([[:space:]]*dimreduce:[[:space:]]*".*")/\1/; /^encoders:/,/^\]/{s/^([[:space:]]*)"resnet50"(,?)/\1# "resnet50"\2/}' ./HEST/bench_config/bench_config.yaml
```

Then finally run our HEST_evaluation.py script to benchmark your checkpoint. Running HEST_evaluation.py will automatically download the necessary preprocessed patches and patch encoders and output results into a new `path-fm-dinov3/eval` folder (specifically, the benchmark results dump to the `ST_pred_results/` subfolder).

```bash
python HEST_evaluation.py --checkpoint <path_to_your_eval_checkpoint>
```

## Related Work / Citation

This repository adapts and extends Meta AI's Dinov3 codebase and follows modifications introduced by Kaiko's Midnight work. If you use this repository or models in academic work, please cite their and our work:

Kaplan, D., Grandhi, R. S., Lane, C., Warner, B., Abraham, T. M., & Scotti, P. S. (2025). How to train a state-of-the-art pathology foundation model with $1.6k. *Sophont*. https://sophont.med/blog/path-fm-dinov3

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., ... & Bojanowski, P. (2023). Dinov3: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193.

Karasikov, M., van Doorn, J., Känzig, N., Erdal Cesur, M., Horlings, H. M., Berke, R., ... & Otálora, S. (2025). Training state-of-the-art pathology foundation models with orders of magnitude less data. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 573-583). Cham: Springer Nature Switzerland.
