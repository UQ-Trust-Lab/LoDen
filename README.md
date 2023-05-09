# LoDen
##Citation
If you use Loden for research, please cite the following paper:

@article{Ma2023LoDen,
title={LoDen: Making Every Client in Federated Learning a Defender Against the Poisoning Membership Inference Attacks},
author={Ma, Mengyao and Zhang, Yanjun and Arachchige, Pathum Chamikara Mahawaga and Zhang, Leo Yu and Baruwal Chhetri, Mohan and Bai, Guangdong},
booktitle={18th ACM ASIA Conference on Computer and Communications Security {ASIACCS 2023'}},
year={2023},
publisher={ACM}
}



## Code structure
To run the experiments, please run *setting*_optimized.py, replace *setting* with the LoDen knowledge settings to experiment with.
Here, there are three key runable files in the repository, including
* loden_blackbox_defence.py  # Blackbox LoDen defence experiment
* loden_whitebox_defence.py # Whitebox based LoDen defence experiment
* loden_MIA.py            # Basic MIA experiment

Other files in the repository
* __constants.py__ Experiment constants, contains the default hyperparameter set for each experiment
* __data_reader.py__ It is used for read dataset, split dataset into training set and test set for each participant
* __aggregator.py__ Server-side robust aggregators, including
  * Fang [1] 
  * Median [2] 
  * Trimmed-Mean [2]
  * FLTrust [3]
  * Multi-Krum [4]

* __models.py__ The models including target and attack model are wrapped in this file, and also including
  * Basic target and attack models
  * MIA algorithm

## Instructions for running the experiments
### 1. Set the experiment parameters
The experiment parameters are defined in __constants.py__. To execute the experiments, please set the parameters in __constants.py__ file. 

### 2. Run the experiment
To run the experiment, please run loden_*setting*_defence.py, replace *setting* with the LoDen knowledge settings to experiment with. You can use command line to run the experiment, e.g. in a LINUX environment, to execute the *Black-box LoDen* experiment, please input the following command under the source code path

```python loden_blackbox_defence.py```

To execute the *whitebox LoDen* experiment, please input the following command under the source code path

```python loden_whitebox_defence.py```

To execute the *MIA baseline* experiment, please input the following command under the source code path

```python loden_MIA.py```

### 3. Save the experiment results
After the experiment is finished, the experiment results will be saved in the directory defined in __constants.py__ . The experiment results include the Local nodes log, Attacker log, LoDen defence log and the FL training log.

## Understanding the output
The result of the experiment will be saved in the directory defined in __constants.py__ (default *output* directory), including
* Local nodes log file (E.g. the log for local node *0*, 2023_04_21_22Location30MedianTrainEpoch100predicted_vector_round0100optimized_model_single_honest_side_defence_0.csv): This file contains the training samples log, including:
  * class prediction: the class prediction of the sample
  * class label: the class label of the sample
  * the confidence vector of groundtruth class: the probability of the groundtruth class prediction
  * sample index: the index of the sample

* Attacker log file (E.g.2023_04_21_22Location30MedianTrainEpoch100predicted_vector_round0100optimized_model_single_defence.csv): This file contains the victim samples log, including:
  * class prediction: the class prediction of the sample
  * class label: the class label of the sample
  * the confidence vector of groundtruth class: the probability of the groundtruth class prediction
  * sample index: the index of the sample
  
* LoDen defence log file (E.g.2023_04_21_22Location30MedianTrainEpoch100predicted_vector_round0100optimized_model_single_honest_side_defence_removed.csv): This file contains the local node samples removed by LoDen log, including:
  * sample removed round: the round of the FL training when the sample is removed
  * sample class belongs: the class label of the sample
  * sample index: the index of the sample

* FL training log file (E.g.2023_04_21_22Location30MedianTrainEpoch100AttackEpoch100honest_node_round0optimized_model_single_ascent_factor15_honest_side_defence.txt): This file contains the FL training log, including:
  * round: the round of the FL training
  * test loss: the test loss of the global model for current round
  * test accuracy: the test accuracy of the global model for current round
  * train accuracy: the train accuracy of the global model for current round
  * time: the current time of the FL training


## Requirements
Recommended to run with conda virtual environment
* Python 3.10.5
* PyTorch 1.13.0
* numpy 1.22.4
* pandas 1.4.2

## Reference
[1] Minghong Fang, Xiaoyu Cao, Jinyuan Jia, and Neil Gong. 2020. Local model poisoning attacks to byzantine-robust federated learning. In 29th {USENIX} Security Symposium ( {USENIX } Security 20). 1605–1622.

[2] Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. 2018. Byzantine-robust distributed learning: Towards optimal statistical rates. In Inter- 1246 national Conference on Machine Learning. PMLR, 5650–5659.

[3] Xiaoyu Cao, Minghong Fang, Jia Liu, Neil Zhenqiang Gong. FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping. In 28th Network and Distributed Systems Security (NDSS) Symposium 2021

[4] Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer.2017. Machine learning with adversaries: Byzantine tolerant gradient descent. In  Proceedings of the 31st International Conference on Neural Information Processing Systems

