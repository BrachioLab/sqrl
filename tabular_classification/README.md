# Code of SQRL for tabular classification task
## Configurations
We prepared a docker file in the folder "docker_files/". To use it, we first of all copy the docker file for the tabular classification task to the root directory of this project (assuming it is "/path/to/proj/"):
```
cd /path/to/proj/
cp docker_files/Dockerfile_tabular_classification Dockerfile
```

Then we initialize a docker container for reproducing our experimental results with the following command:
```
docker build -t tabular_classification .
```

one way of entering the docker container is using the following command:
```
docker run  -v /path/to/data/:/data/:Z -it tabular_classification
```

Then within this container, we use the following command to initialize the cuda configurations:
```
apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub && apt-get update && apt-get -y install cuda
```

There will be some error messages when the above command is executed. But it won't influenece the following experiments. 

In the end, the following command is needed to install extra python libraries:
```
cd tabular_classification/
pip install -r requirements.txt
```

## Prepare datasets:
* Download cardiovascular dataset from [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset). Extract the zip file and put the csv file "cardio\_train.csv" to a folder "/path/to/data/"

## Perform data preprocessing operation with the following command:
```
mkdir /path/to/data/output/
python pre_processing.py --work_dir /path/to/data/output/ --data_dir /path/to/data/
```
This will split the original csv file into three parts, i.e., "C.csv", "N.csv" aand "y.csv", which store the categorical features, numerical features and labels respectively.

## Adjust the configuration file with your customized data path:
```
sed -i.backup  's#/path/#/path/to/data/#g' config/dataset/cardio_upstream.yaml
sed -i.backup  's#/path/#/path/to/data/#g' config/dataset/cardio_downstream.yaml
```



## Convert numerical features into categorical features over which the statistical rules are learned:
```
python generate_rule_learning_db.py --work_dir /path/to/data/output/ --data_dir /path/to/data/ 
```

We prepared the rules that we generated for the experiments, which are stored in "tabular_classification/rule_after_validations/". You can just copy those rules to the directory "/path/to/data/" so that you can skip the following two steps for learning and validating statistical rules, i.e.:
```
cp rule_after_validations/* /path/to/data/
```

Otherwise, you can go through the following two steps to create rules from scratch:

## Learn statistical rules with the following command:
```
TO BE DONE
```
The code for the rule learning component of the SQRL framework is being finalized and will be released soon.
For now, we provide the learned rules for the tabular classification task in the folder `rules_after_validations/`.

## Validating rules by removing rules that are not generalized well from training set to validation set:
```
python cardio_transformer.py model=ft_transformer_pretrain dataset=cardio_upstream hyp=hyp_for_test_time_adaptation_rule_eval
```




## Do training without test-time adaptation with the following command:
```
python cardio_transformer.py model=ft_transformer_pretrain dataset=cardio_upstream
```
This will result in a learned model stored in the path "/path/to/data/output/ft\_transformer/"


## Perform test-time adaptations with the following command:
```
python cardio_transformer.py model=ft_transformer_pretrain dataset=cardio_downstream hyp=hyp_for_test_time_adaptation_$method
```

This will load the learned model stored in  "/path/to/data/output/ft\_transformer/" and perform test-time adaptation with a certain test-time adaptation method, in which the choice of $Method is as follows:
$method=rule: our method

$method=cpl: conjugate pseudo-label [[2]](#2)

$method=rpl: Robust Pseudo-label [[3]](#3)

$method=tent: Tent [[4]](#4)

$method=norm: Batch normalization [[5]](#5)

## References
<a id="1">[1]</a>
Du, Wenjie, David Côté, and Yan Liu. "Saits: Self-attention-based imputation for time series." Expert Systems with Applications 219 (2023): 119619.

<a id="2">[2]</a>
Goyal, S., Sun, M., Raghunathan, A., and Kolter, J. Z. Test time adaptation via conjugate pseudo-labels. In Advances in Neural Information Processing Systems, 2022.

<a id="3">[3]</a>
Rusak, E., Schneider, S., Pachitariu, G., Eck, L., Gehler, P. V., Bringmann, O., Brendel, W., and Bethge, M. If your data distribution shifts, use self-learning. Transactions of Machine Learning Research, 2021.

<a id="4">[4]</a>
Wang, D., Shelhamer, E., Liu, S., Olshausen, B., and Darrell, T. Tent: Fully test-time adaptation by entropy minimization. In International Conference on Learning Representations, 2020.

<a id="5">[5]</a>
Ioffe, S. and Szegedy, C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning, pp. 448–456. PMLR, 2015. 
