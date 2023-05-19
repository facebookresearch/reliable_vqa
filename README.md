# Reliable VQA

This is the implementation for the ECCV 2022 paper [Reliable Visual Question Answering: Abstain Rather Than Answer Incorrectly](https://arxiv.org/abs/2204.13631). If you find our paper or this repository useful for your own work, please cite:
```
@inproceedings{whitehead2022reliablevqa,
  title={Reliable Visual Question Answering: Abstain Rather Than Answer Incorrectly},
  author={Whitehead, Spencer and Petryk, Suzanne and Shakib, Vedaad and Gonzalez, Joseph and Darrell, Trevor and Rohrbach, Anna and Rohrbach, Marcus},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

This repository uses [PyTorch](https://pytorch.org/) and is built on top of [MMF](https://mmf.sh/). It contains the following:
- Implementations for the metrics used in our paper, including risk, coverage, and Effective Reliability.
- Implementations for maximum probability (MaxProb), calibration, and learned multimodal selection functions (Selector).
- Training configs for the models in our work.
- Download links for the VQA v2 dataset splits, trained model checkpoints, and pre-extracted features used in our work.

**Update:** The previous version of this repo unintentionally used the default ground truth answers in MMF in the field `answers` which has slightly different reference answers (e.g., it replaces some entries if they are not in the vocabulary of 3k answers). Therefore, we provide a standalone evaluation script ([eval_scripts/](eval_scripts/)) that uses the original VQA v2 annotations as references for evaluations. This update is reflected in the arXiv version (please see Changelog in the appendix) and future work should use the updated evaluations for reporting results.


## Repo Organization

The folders in this repo are structured as follows:

- `configs/`:
    - `experiments/` contains YAML configs to train each of the VQA models and corresponding selection functions.
    - `datasets/` contains the YAML config for the custom `vqa2_extended` dataset.
- `datasets/`: contains the dataset implementation and builder for `vqa2_extended`. This dataset is the same as VQA v2 within MMF, but it also supports additional model confidence outputs for the selection functions and multiple choice annotations for calibration evaluation.
- `eval_scripts/`: contains evaluation scripts for computing risk-coverage and Effective Reliability metrics in the paper.
    - `reliable_vqa_eval.py`: contains an evaluator object that computes the different metrics.
    - `run.py`: script for running the evaluations given model predictions and the ground truth annotations.
    - `vqa.py`: contains an object for interfacing with the VQA v2 annotations and predictions.
- `models/`: for each VQA model in our experiments, we register a version on top of the original model which returns additional confidence and intermediate feature outputs needed for the selection functions.
    - `selective_predictors.py` contains implementations for both calibration and Selector models.
- `modules/`:
    - `losses.py` contains the correctness-prediction loss function for learned Selector models.
    - `metrics.py` contains implementations of the risk-coverage and Effective Reliability metrics in MMF for validation.
- `__init__.py`: imports custom MMF components to be used by MMF.


## Environment Setup
Please follow the MMF installation instructions here: https://mmf.sh/docs/. We recommend installing from source. Note, when installing from source, you do not need to clone the MMF repository under this repo. You can simply clone MMF to its own directory. We also recommend using a conda environment for the installation and running, which can be used for both MMF and this repo.

Following the MMF installation, your environment should have Python 3.7+ and PyTorch 1.6+ installed. You will also need scikit-learn 1.0+ and pandas 1.3.4+.


## Data Setup

**TL;DR:** We use the [VQA v2](https://visualqa.org/) dataset. We split the VQA v2 validation set into 3 parts and provide the annotations below. We also extract custom grid features for the CLIP-ViL model, provided below. All other annotations and features are automatically downloaded by MMF, as specified by each of the configs in this repo.

### Downloading Annotations and Features

First, download the original VQA v2 validation question and answer annotation JSON files from here: https://visualqa.org/download.html. These will be used for the evaluations.

When running MMF with one of our config files for the first time, MMF should automatically download the features and annotations for VQA v2. These directories/files will be stored in the `$MMF_DATA_DIR` (`env.data_dir`) under a `vqa2` directory. Please see MMF for more details on this. We recommend starting by running Pythia+MaxProb through this repo, which will download the annotations and features used for Pythia, ViLBERT, and VisualBERT (see [Training](#training) for details)

We also recommend saving our validation splits and CLIP features (described in the next sections) within these directories as well, and the following setup assumes this is the case. If you decide to structure your directories differently, then you will need to update your config path, etc. accordingly.

### Custom VQA v2 Validation Splits

The standard VQA v2 training set is used for training VQA models. However, since answer annotations are not available for the test-dev and test-std VQA v2 splits, we split the VQA v2 validation set into 3 disjoint sets (i.e., no images or questions are shared) for evaluation purposes:
- `dev`: validation set for the VQA model training and training set for the selective predictors.
- `val`: validation set for the selective predictors.
- `test`: test set for all models, and what we report results on in our paper.

These split annotation files can be downloaded here: [download](https://dl.fbaipublicfiles.com/reliablevqa/data/reliable_vqa-annotations.tar.gz)

Once downloaded, place the compressed file in the `<env.data_dir>/vqa2/` directory. Decompressing the file should will set up the following directory structure:
```
vqa2/
    reliable_vqa/
        annotations/
            imdb_val2014-dev.npy
            imdb_val2014-test.npy
            imdb_val2014-val.npy
```

To use our config files as is, these annotation files should be placed under the path `<env.data_dir>/vqa2/reliable_vqa/annotations`. Otherwise, you will need to edit the config and annotation files to match your paths. For instance, the dataset annotations in a config for training a VQA model are:
```
dataset_config:
  vqa2_extended:
    annotations:
      train:
      - vqa2/defaults/annotations/imdb_train2014.npy
      val:
      - vqa2/reliable_vqa/annotations/imdb_val2014-dev.npy
      test:
      - vqa2/reliable_vqa/annotations/imdb_val2014-test.npy
```

Whereas the annotations for training a Selector are:
```
dataset_config:
  vqa2_extended:
    annotations:
      train:
      - vqa2/reliable_vqa/annotations/imdb_val2014-dev.npy
      val:
      - vqa2/reliable_vqa/annotations/imdb_val2014-val.npy
      test:
      - vqa2/reliable_vqa/annotations/imdb_val2014-test.npy
```

### CLIP-ViL Grid features

For training all VQA models, we use pre-extracted features instead of images for speed and consistency. The Pythia, ViLBERT, and VisualBERT models all use features which can be downloaded automatically upon running via MMF. However, CLIP-ViL uses grid image features from [CLIP](https://github.com/openai/CLIP). We provide our pre-computed features as well as a slightly adjusted version of the extraction script from the [CLIP-ViL repo](https://github.com/clip-vil/CLIP-ViL/tree/master/CLIP-ViL-Direct/vqa) that can be used to extract CLIP features independently.

#### Pre-extracted Features

1. Download the features ([download](https://dl.fbaipublicfiles.com/reliablevqa/data/clip_features.tar.gz)) and their updated annotation files ([download](https://dl.fbaipublicfiles.com/reliablevqa/data/reliable_vqa-clip-annotations.tar.gz)). IMPORTANT: The features are very large (~150GB when compressed) and may take a long time to download.
2. Decompress the annotations inside the file in `<env.data_dir>/vqa2/`, which yields:
    ```
    vqa2/
        reliable_vqa-clip/
            annotations/
                imdb_train2014.npy
                imdb_val2014-dev.npy
                imdb_val2014-test.npy
                imdb_val2014-val.npy
    ```
3. Place the downloaded features in a directory alongside the annotations directory:
    ```
    vqa2/
        reliable_vqa-clip/
            annotations/
                ...
            clip_features.tar.gz
    ```
4. Decompress the features within the `reliable_vqa-clip` directory. Your directory structure should mirror MMF's:
    ```
    vqa2/
        reliable_vqa-clip/
            annotations/
                ...
            features/
                train2014/
                val2014/
    ```

#### [OPTIONAL] Extracting Your Own Features

0. \[Optional\] We recommend creating a separate conda environment (with Python 3.7+) for the feature extraction.
1. Clone the [CLIP-ViL repo](https://github.com/clip-vil/CLIP-ViL) and follow their installation/setup instructions (i.e., install [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) from the CLIP-ViL provided local clone). Note that the CLIP-ViL repo does not need to be cloned within this repo.
2. Download the [COCO train+val 2014 images and annotations](https://cocodataset.org/#download) and place them in a directory with the following structure and path names:
    ```
    coco_2014/
        annotations/
            instances_train2014.json
            instances_val2014.json
        images/
            train2014/
            val2014/
    ```
2. Copy/move `fixed_mcan_clip_grid_feature.py` to `CLIP-ViL/CLIP-ViL-Direct/vqa` in the CLIP-ViL repo.
3. Change `OUTPUT_DIR` in `CLIP-ViL/CLIP-ViL-Direct/vqa/configs/R-50-grid.yaml` to your desired directory for the features (i.e., `<env.data_dir>/vqa2/reliable_vqa-clip/features`).
4. Run the following on the train2014 images (repeat using `coco_2014_val` to run on val2014 images):
    ```
    DETECTRON2_DATASETS=<PATH_TO_PARENT_DIR_OF_coco_2014> python fixed_mcan_clip_grid_feature.py --config-file configs/R-50-grid.yaml --dataset coco_2014_train --model_type RN50x4
    ```
5. You can download the updated annotations following the [Pre-extracted features section](#pre-extracted-features) or you can run the mapping script to create updated annotation files for CLIP-ViL:
    ```
    python clipvil_anns_conversion.py --input_dir <env.data_dir>/vqa2/reliable_vqa/annotations --output_dir <env.data_dir>/vqa2/reliable_vqa-clip/annotations
    ```


## Model Checkpoints

We provide trained model checkpoints for each combination of the 4 VQA models and 3 selection functions in our paper. Note that the MaxProb model checkpoints are simply the VQA models. The Calibration and Selector selective predictors themselves are much smaller than the VQA models, yet we include the VQA model in their corresponding checkpoints for convenience.

Note, the MaxProb ViLBERT and VisualBERT are the same as those from MMF (pre-trained and fine-tuned), so they can also be downloaded via the MMF model zoo. From the MMF model zoo, ViLBERT corresponds to [`vilbert.finetuned.vqa2.from_vqa2_train`](https://github.com/facebookresearch/mmf/tree/main/projects/pretrain_vl_right#vilbert-finetuned-models) and VisualBERT corresponds to [`visual_bert.finetuned.vqa2.from_coco_train`](https://github.com/facebookresearch/mmf/tree/main/projects/pretrain_vl_right#visualbert-finetuned-models).

<table>
    <thead>
        <tr>
            <th></th>
            <th>MaxProb</th>
            <th>Calibration</th>
            <th>Selector</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Pythia</td>
            <td align="center"><a href="https://dl.fbaipublicfiles.com/reliablevqa/models/maxprob_pythia.tar.gz">download</a></td>
            <td align="center"><a href="https://dl.fbaipublicfiles.com/reliablevqa/models/calibration_pythia.tar.gz">download</a></td>
            <td align="center"><a href="https://dl.fbaipublicfiles.com/reliablevqa/models/selectpred_pythia.tar.gz">download</a></td>
        </tr>
        <tr>
            <td>ViLBERT</td>
            <td align="center"><a href="https://dl.fbaipublicfiles.com/mmf/data/models/vilbert/vilbert.finetuned.vqa2.train.tar.gz">MMF download</a></td>
            <td align="center"><a href="https://dl.fbaipublicfiles.com/reliablevqa/models/calibration_vilbert.tar.gz">download</a></td>
            <td align="center"><a href="https://dl.fbaipublicfiles.com/reliablevqa/models/selectpred_vilbert.tar.gz">download</a></td>
        </tr>
        <tr>
            <td>VisualBERT</td>
            <td align="center"><a href="https://dl.fbaipublicfiles.com/mmf/data/models/visual_bert/visual_bert.finetuned.vqa2.train.tar.gz">MMF download</a></td>
            <td align="center"><a href="https://dl.fbaipublicfiles.com/reliablevqa/models/calibration_visual_bert.tar.gz">download</a></td>
            <td align="center"><a href="https://dl.fbaipublicfiles.com/reliablevqa/models/selectpred_visual_bert.tar.gz">download</a></td>
        </tr>
        <tr>
            <td>CLIP-ViL</td>
            <td align="center"><a href="https://dl.fbaipublicfiles.com/reliablevqa/models/maxprob_movie_mcan.tar.gz">download</a></td>
            <td align="center"><a href="https://dl.fbaipublicfiles.com/reliablevqa/models/calibration_movie_mcan.tar.gz">download</a></td>
            <td align="center"><a href="https://dl.fbaipublicfiles.com/reliablevqa/models/selectpred_movie_mcan.tar.gz">download</a></td>
        </tr>
    </tbody>
</table>

## Sample Commands

Here, we provide sample commands for training and evaluating models. These examples use the CLIP-ViL model (referred to as `movie_mcan`, which is the corresponding model architecture). Running with other models simply involves changing the `config` to the correct path, and changing the `model` argument to one of `pythia`, `vilbert`, `visual_bert` or `movie_mcan` (when using MaxProb) or using `select_*` for a model `*` (when using Calibration or a Selector, e.g., `select_visual_bert`). Note that the annotation files for CLIP-ViL are different because CLIP features are used (see, e.g., `configs/experiments/movie_mcan/vqa2/defaults.yaml`), while all other models use the same set of annotation files, so be sure to use the correct corresponding annotation files and feature paths.

All commands should be ran from the `reliable_vqa` directory, and set `env.user_dir=<PATH_TO_REPO>/reliable_vqa` in the MMF command line options (or, equivalently, `MMF_USER_DIR=$PWD` before the command).

### Training

To train a VQA model:
```
mmf_run env.user_dir=<PATH_TO_REPO>/reliable_vqa env.data_dir=<YOUR_DATA_DIR> env.save_dir=<YOUR_MAXPROB_SAVE_DIR> dataset=vqa2_extended model=movie_mcan config=configs/experiments/movie_mcan/vqa2/defaults.yaml run_type=train_val
```

To train a learned multimodal selection function (Selector) for the VQA model:
```
mmf_run env.user_dir=<PATH_TO_REPO>/reliable_vqa env.data_dir=<YOUR_DATA_DIR> env.save_dir=<YOUR_SELECTOR_SAVE_DIR> dataset=vqa2_extended model=select_movie_mcan config=configs/experiments/movie_mcan/vqa2/select_pred.yaml run_type=train_val checkpoint.resume_pretrained=True checkpoint.resume_file=<YOUR_MAXPROB_SAVE_DIR>/best.ckpt
```
The `checkpoint.resume_file` option could also be one of the `model.pth` files downloaded above. Also, it's best to make sure that the `env.save_dir` for MaxProb and Selector are different. Otherwise, they will overwrite each other.

For ViLBERT and VisualBERT, we utilize the models already fine-tuned on VQA v2 that are provided by MMF. These serve as our MaxProb selective models for ViLBERT and VisualBERT. To train the Selector with ViLBERT or VisualBERT, you should provide the `checkpoint.resume_file` path to the MMF model `.pth` file downloaded from the model zoo (or the link above):
```
mmf_run env.user_dir=<PATH_TO_REPO>/reliable_vqa env.data_dir=<YOUR_DATA_DIR> env.save_dir=<YOUR_SELECTOR_SAVE_DIR> dataset=vqa2_extended model=select_visual_bert config=configs/experiments/visual_bert/vqa2/select_pred.yaml run_type=train_val checkpoint.resume_pretrained=True checkpoint.resume_file=<YOUR_MMF_MODEL_SAVE_DIR>/visual_bert.finetuned.vqa2.from_coco_train/model.pth
```

### Evaluation

We first make predictions on the val and test sets, and then evaluate these using the evaluation scripts.

To get predictions, change the run type to test (`run_type=test`), add the argument `evaluation.predict=True`, and replace the `test` annotation path in the config with that of the annotations to get predictions on (e.g., `vqa2/reliable_vqa/annotations/imdb_val2014-test.npy`, `vqa2/reliable_vqa/annotations/imdb_val2014-val.npy`):
```
mmf_run env.user_dir=<PATH_TO_REPO>/reliable_vqa env.data_dir=<YOUR_DATA_DIR> env.save_dir=<YOUR_RESULT_SAVE_DIR> dataset=vqa2_extended model=select_movie_mcan config=configs/experiments/movie_mcan/vqa2/select_pred.yaml run_type=test evaluation.predict=True checkpoint.resume=True checkpoint.resume_file=<YOUR_SELECTOR_SAVE_DIR>/best.ckpt dataset_config.vqa2_extended.annotations.test=vqa2/reliable_vqa-clip/annotations/imdb_val2014-test.npy
```
For getting predictions from ViLBERT and VisualBERT with MaxProb, you can also simply use the model zoo versions of these:
```
mmf_run env.user_dir=<PATH_TO_REPO>/reliable_vqa env.data_dir=<YOUR_DATA_DIR> env.save_dir=<YOUR_RESULT_SAVE_DIR> dataset=vqa2_extended model=visual_bert config=configs/experiments/visual_bert/vqa2/defaults.yaml run_type=test evaluation.predict=True checkpoint.resume=True checkpoint.resume_zoo=visual_bert.finetuned.vqa2.from_coco_train dataset_config.vqa2_extended.annotations.test=vqa2/reliable_vqa-clip/annotations/imdb_val2014-test.npy
```
This will produce a JSON file (similar format to the [VQA v2 result format](https://visualqa.org/evaluation.html)) within `env.save_dir` containing the model's answers and confidences that we use to evaluate. Repeat this using `imdb_val2014-val.npy` as the test set to get results on the val data for choosing thresholds.

Next, we use a standalone evaluation script to get the evaluation metrics, which accepts the original VQA v2 question and annotation JSONs as references:
```
python eval_scripts/run.py \
--questions <PATH>/v2_OpenEnded_mscoco_val2014_questions.json \
--annotations <PATH/v2_mscoco_val2014_annotations.json \
--predictions <RESULTS_ON_TEST_DATA>.json \
--threshold_predictions <RESULTS_ON_VAL_DATA>.json
```
This command will output **VQA accuracy**, **coverage@risk**, **AUC** for the risk-coverage curve, and **Effective Reliability**. Note, since this uses the original VQA v2 annotations and a similar format to VQA result format, this evaluation script should be compatible with predictions from models outside this repo by simply providing an extra `confidence` field in the predictions.

## Acknowledgements

We would like to thank the creators of MMF for their open-source implementations. We thank Sheng Shen and the authors of [How Much Can CLIP Benefit Vision-and-Language Tasks?](https://arxiv.org/abs/2107.06383) for providing assistance on extracting features and reproducing their model as well as releasing their code. We also thank Aishwarya Agrawal for input on the evaluations. Lastly, we thank Grace Luo for early assistance with MMF.


## License

The majority of Reliable VQA is licensed under CC-BY-NC (see [LICENSE](LICENSE) for details), however [fixed_mcan_clip_grid_feature.py](fixed_mcan_clip_grid_feature.py), which is modified from the [mcan_clip_grid_feature.py](https://github.com/clip-vil/CLIP-ViL/blob/master/CLIP-ViL-Direct/vqa/mcan_clip_grid_feature.py) script in https://github.com/clip-vil/CLIP-ViL/tree/master/CLIP-ViL-Direct/vqa, is licensed under the Apache 2.0 license and [eval_scripts/vqa.py](eval_scripts/vqa.py) as well as [eval_scripts/reliable_vqa_eval.py](eval_scripts/reliable_vqa_eval.py), which are modified from [vqa.py](https://github.com/GT-Vision-Lab/VQA/blob/master/PythonHelperTools/vqaTools/vqa.py) and [vqaEval.py](https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py) in https://github.com/GT-Vision-Lab/VQA, are licensed under the BSD 2-Clause License.
