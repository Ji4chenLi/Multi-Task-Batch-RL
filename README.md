# Multi-Task Batch RL

TODO: Change the name of tiMe.

This repository contains the code accompanying 'Multi-Task Batch Reinforcement Learning', that is submitted to the NeurIPS 2020

The codes to generate the transition batch for each training tasks was by modiyfing the codes as provided in the [oac-explore](https://github.com/microsoft/oac-explore).

The Batch RL part of this paper is based on the codes as provided in the [BCQ](https://github.com/sfujim/BCQ/tree/master/continuous_BCQ).

Codes for the full model algorithm and each of the baseline and ablation can be found under their correponding folder.

# Reproducing Results

To reproduce the results, we provide the collected transition buffers for each of the training tasks, the trained BCQ models and ensemble predictors in the [Google Drive](https://drive.google.com/open?id=1ZNmxYE3Gym2uxSmV5OjAkKRWECrQgez1), i.e., the first phase of training pipeline. Please download all the data and put them in the ```data``` folder. Otherwise you should be careful when running the following experiments and you should corretly specify the locations.

Experiments are configured via `.py` configuration files located in `./configs`. To reproduce an experiment, you can first go to the corresponding folder, and then run the following commands:


```
python main.py --config=DOMAIN_NAME
```

For example, if you would like to reproduce the results of ``Ant-Dir``, using the full model, then you should do 

```
cd tiMe_full_model
python main.py --config=ant-dir
```

# Running Experiments

If you would like to generate these results for the training phase, first you can go to the ``oac-explore`` folder and run the following command to obtain the training buffers. Note that the list of ``goal_id`` is varying from domain to domain:

```
python main.py --config=DOMAIN_NAME --goal=GOAL_ID
```

Then you can go the the ``BCQ`` folder and run te following command to extract task-specific results:

```
python main.py --config=DOMAIN_NAME --goal=GOAL_ID
```

Simultaneously, you can get the model prediction ensembles by going to the ``reward_prediction_ensemble`` and runing

```
python main.py --config=DOMAIN_NAME --goal=GOAL_ID
```

For software dependencies, please have a look inside the ```environment``` folder, you can either build the Dockerfile, create a conda environment with ```environment.yml``` or pip environment with ```environments.txt```.

To create the conda environment, ```cd``` into the ```environment``` folder and run:

```
python install_mujoco.py
conda env create -f environment.yml
```

# Acknowledgement

This reposity was based on [rlkit](https://github.com/vitchyr/rlkit), [oac-explore](https://github.com/microsoft/oac-explore), [BCQ](https://github.com/sfujim/BCQ/tree/master/continuous_BCQ) and [PEARL](https://github.com/katerakelly/oyster).

# Citation

If you use the codebase, please cite the paper:

```
@misc{oac,
    title={Better Exploration with Optimistic Actor-Critic},
    author={Kamil Ciosek and Quan Vuong and Robert Loftin and Katja Hofmann},
    year={2019},
    eprint={1910.12807},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
```

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.