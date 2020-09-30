# Multi-Task Batch RL with Metric Learning

This repository contains the code accompanying the paper ```Multi-Task Batch Reinforcement Learning with Metric Learning```.

Codes for the full model algorithm and each of the baseline and ablation can be found under their corresponding folder. For ease of use, we separate each method into a single folder. Thus, there are duplicate files in this repository.

# Software dependencies 

For software dependencies, please have a look inside the ```environment``` folder, one can create a conda environment with ```environment.yml```.

To create the conda environment, ```cd``` into the ```environment``` folder and run:

```
python install_mujoco.py
conda env create -f environment.yml
```

** To reproduce the results of ``Fig. 5`` in our paper ([MetaGenRL](https://github.com/louiskirsch/metagenrl) related), one should create another conda environment and run the following commands:

```
pip3 install ray[tune]==0.7.7 gym[all] mujoco_py>=2 tensorflow-gpu==1.15.2 scipy numpy

python3 -c 'import ray; from pyarrow import plasma as plasma; plasma.build_plasma_tensorflow_op()'

```
which is slightly different from the instruction in the official [MetaGenRL](https://github.com/louiskirsch/metagenrl) repo.

# Data to download 

To reproduce the results, we provide the collected transition buffers for each of the training tasks, the trained BCQ models and ensemble predictors in the [Google Drive](https://drive.google.com/file/d/1YqskGjcPURHs-Al3wGs4ddVKBcw6np5q/view?usp=sharing), i.e., the first phase of training pipeline. Please download all the data and put them in the ```data_and_trained_models``` folder. Otherwise one should be careful when running the following experiments and one should correctly specify the locations. After downloading the files from Google Drive, use the command below to unzip the file:

```
tar -xvf data_and_trained_model.tar.gz
```

To reproduce the results of Fig. 11, one should further download the multi-task policy trained without reward ensemble from the [link](https://drive.google.com/file/d/1zIGOt3DgqtOqFjnKzVOKL5L92hwiOmra/view?usp=sharing). After downloading the file, use the following command to unzip the files
```
tar -xvf full_model_no_ensemble_results.tar.gz
```
then move the downloaded folder to the ```data_and_trained_models``` folder.
# Reproducing Results


Experiments are configured via `.py` configuration files located in `./configs`. To evaluate different methods, we take ``AntDir`` as an example. Except for ``WalkerParam``, one can obtain the corresponding results of the other task distributions by simply changing the distribution name. We will specify the difference for ``WalkerParam`` and provide the codes to reproduce the results.

## Reproducing Results ``Fig. 4`` in our paper.

1. To get the results of our model, one should go to the ``full_model`` folder and run the experiments:

    ```
    cd full_model
    python main.py --config=ant-dir
    ```
2. Note that to get the results of our model on ``WalkerParam``, one should go to the ``full_model_walker_param`` folder instead of ``full_model``, i.e.

    ```
    cd full_model_walker_param
    python main.py --config=walker-param
3. Similarly, to get the results of ``Contextual BCQ``, one should go to the ``contextual_bcq`` folder and run the experiments:

    ```
    cd contextual_bcq
    python main.py --config=ant-dir
    ```
4. To get the results of ``PEARL``, one should go to the ``batch_pearl`` folder and run the experiments:

    ```
    cd batch_pearl
    python launch_experiment.py './configs/ant-dir.json'
    ```
5. After you repeat the procedures above for all the 6 task distributions, one can plot the results by running 

    ```
    python -m plotting.evaluate_against_baseline
    ```

## Reproducing Results ``Fig. 5`` in our paper.
Note that to reproduce the results, one need to activate the correct conda environment

1. Run the codes to obtain the training results of MetaGenRL on ``AntDir``
    ```
    cd metagenrl
    python ray_experiments.py train
    ```

2. Open the tensorboard using the following comments:
    ```
    tensorboard --logdir ./ray_results/metagenrl
    ```

3. Download the results of ``custom_mean_episode_reward`` in tensorboard of either agent ``0`` or agent ``1``. Save the file as ``custom_mean_episode_reward.csv``.

4. Plot the results by running
    ```
    python -m plotting.evaluate_metagenrl
    ```

## Reproducing Results ``Fig. 6`` in our paper.

 Note that we already obtain the results of our full model by following the procedures to reproduce Fig. 4 in our paper.

1. To get the results of ``No transition relabelling``, one should go to the ``no_transition_relabelling`` folder and run the experiments

    ```
    cd no_transition_relabelling
    python main.py --config=ant-dir
    ```
2. To get the results of ``No triplet loss``, one should go to the ``no_triplet_loss`` folder and run the experiments

    ```
    cd no_triplet_loss
    python main.py --config=ant-dir
    ```
3. Note that to get the results of ``No triplet loss`` on ``WalkerParam``, one should go to the ``no_triplet_loss_walker_param`` folder instead of ``no_triplet_loss``, i.e.

    ```
    cd no_triplet_loss_walker_param
    python main.py --config=walker-param
    ```
4. To get the results of ``neither``, one should go to the ``neither`` folder and run the experiments:

    ```
    cd neither
    python main.py --config=ant-dir
    ```
5. For UmazeGoal-M, we also need the results of ``GT``, one should go to the ``full_model_ground_truth_label`` folder and run the experiments:
    ```
    cd full_model_ground_truth_label
    python main.py --config=maze-umaze
    ```
6. After you repeat the procedures above for all the 6 task distributions, one can plot the results by running 

    ```
    python -m plotting.evaluate_against_ablations
    ```

## Reproducing Results ``Fig. 8`` and ``Fig. 13`` in our paper.

To reproduce the results of Fig. 8 and Fig. 13 in the paper, one needs to run the original SAC ``oac-explore`` on ``all`` training tasks. And run the SAC initialized by our method ``sac_with_initialization`` and a variation of SAC ``sac_baseline`` with two identically initialized Q functions trained by different mini-batches on ``all`` testing tasks.

There are 8 testing tasks for all the task distributions with ``GOAL_ID`` ranging from ``1 to 8``. However, the lists of ``GOAL_ID`` of training tasks vary from task distributions to task distributions, which are specified below:

- AntDir: ``[0, 1, 4, 10, 12, 14, 17, 21, 26, 27]``
- AntGoal: ``range from 0 to 9``
- HumanoidDir-M: ``[0, 1, 4, 10, 12, 14, 17, 21, 26, 27]``
- HalfCheetahVel: ``[3, 5, 8, 15, 16, 17, 23, 24, 29, 31]``
- WalkerParam: ``range from 0 to 29``
- UmazeGoal-M: ``range from 0 to 9``

1. To get the results of original SAC, one should go to ``oac-explore`` and run the experiments on all training tasks:

    ```
    cd oac-explore
    python main.py --config=ant-dir --goal=GOAL_ID
    ```
    Note one should repeat experiments with ``GOAL_ID`` traversing ``[0, 1, 4, 10, 12, 14, 17, 21, 26, 27]``.

2. To get the results of ``SAC init by our method``, one should go to the ``sac_with_initialization`` folder and run the experiments on all testing tasks by varying ``GOAL_ID`` from 1 to 8:
    ```
    cd sac_with_initialization
    python main.py --config=ant-dir --goal=GOAL_ID
    ```
3. To get the results of the variation of SAC with two identically initialized Q functions trained by different mini-batches, one should go to the ``sac_baseline`` folder and run the experiments on all testing tasks by varying ``GOAL_ID`` from 1 to 8:
    ```
    cd sac_baseline
    python main.py --config=ant-dir --goal=GOAL_ID
    ```
4. After you repeat the procedures above for all the 6 task distributions, one can plot the results by running 

    ```
    python -m plotting.evaluate_sac_init
    ```

## Reproducing Results ``Fig. 11`` in our paper.

The configuration files are listed in ``full_model/configs`` and ``full_model_walker_param/configs``. File name specified the value of ``triplet margin``. In paper, we set the values of ``triplet margin`` to be ``[0.0, 2.0, 4.0, 8.0]`` and show the results on five task distributions except for ``HalfCheetahVel``.

1. For example, to get the results of ``triplet margin = 0.0``, one can cd to ``full_model`` and run the experiment:
    ```
    cd full_model
    python main.py --config=ant-dir-triplet-margin-0p0
    ```
2. To get the results on ``Walker-Param``, one should go to ``full_model_walker_param`` instead:
    ```
    cd full_model
    python main.py --config=walker-param-triplet-margin-0p0
    ```
3. After you repeat the procedures above for all the 5 task distributions, one can plot the results by running 

    ```
    python -m plotting.evaluate_ablate_triplet_margin
    ```

## Reproducing Results ``Fig. 12`` in our paper.

1.  To obtain the results of SAC initialized by this policy, one should go to the ``sac_with_initialization`` folder and run the experiments on all testing tasks by varying ``GOAL_ID`` from 1 to 8:
    ```
    cd sac_with_initialization
    python main.py --config=humanoid-openai-dir --goal=GOAL_ID  --model_root=../data_and_trained_models/full_model_no_ensemble_results  --base_log_dir=./data_without_ensemble
    ```
2. Note that we have already obtained the results of SAC initialized by our full model and standard SAC when trying to reproduce Fig. 8 and Fig. 13 in the paper. After finishing running the command above for all the testing tasks, one can reproduce Fig. 11 by running 

    ```
    python -m plotting.evaluate_ablate_reward_ensemble
    ```
# Miscellaneous

If you would like to generate the results for the training phase, first you can go to the ``oac-explore`` folder and run the following command to obtain the training buffers. Note that the list of ``GOAL_ID`` varies in different task distributions. The lists of training ``GOAL_ID`` for different task distributions are detailed above when we describe how to reproduce the results of Fig. 8 and Fig. 13 in the paper.

```
python main.py --config=ant-dir --goal=GOAL_ID
```

Then one can go the the ``BCQ`` folder and run the following command to extract task-specific results:

```
python main.py --config=ant-dir --goal=GOAL_ID
```

Simultaneously, one can get the reward prediction ensembles by going to the ``reward_prediction_ensemble`` and running

```
python main.py --config=ant-dir
```

and can get the next state prediction ensembles by going to the ``transition_prediction_ensemble`` and running

```
python main.py --config=walker-param
```

Note that one should pay extra attention to ``--data_models_root``.


# Acknowledgement

This repository was based on [rlkit](https://github.com/vitchyr/rlkit), [oac-explore](https://github.com/microsoft/oac-explore), [BCQ](https://github.com/sfujim/BCQ/tree/master/continuous_BCQ), [PEARL](https://github.com/katerakelly/oyster) and [MetaGenRL](https://github.com/louiskirsch/metagenrl).

The codes to generate the transition batch for each task and codes to accelerate conventional SAC are obtained modifying the codes as provided in the [oac-explore](https://github.com/microsoft/oac-explore).

The Batch RL part of this paper is based on the codes as provided in the [BCQ](https://github.com/sfujim/BCQ/tree/master/continuous_BCQ).

The codes for `batch_pearl` are obtained by modifying [PEARL](https://github.com/katerakelly/oyster).

The codes for `metagenrl` are obtained by modifying [MetaGenRL](https://github.com/louiskirsch/metagenrl).

The codes for each environment files in the folder ``env`` are adapted from [PEARL](https://github.com/katerakelly/oyster). Note that the ``rand_param_envs`` in each folder is copied from [rand_param_envs](https://github.com/dennisl88/rand_param_envs/tree/4d1529d61ca0d65ed4bd9207b108d4a4662a4da0).
