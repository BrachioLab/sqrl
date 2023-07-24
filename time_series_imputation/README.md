# Code of SQRL for time series imputation task

## Configurations
We prepared a docker file in the folder "docker_files/". To use it, we first of all copy the docker file for the time series imputation task to the root directory of this project (assuming it is "/path/to/proj/"):
```
cd /path/to/proj/
cp docker_files/Dockerfile_time_series Dockerfile
```

Then we initialize a docker container for reproducing our experimental results with the following command:
```
docker build -t time_series .
```

Then within this container, we use the following command to initialize the cuda configurations:
```
apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub && apt-get update && apt-get -y install cuda
```

There will be some error messages when the above command is executed. But it won't influenece the following experiments. 

In the end, the following command is needed to install extra python libraries:
```
cd time_series/
pip install -r requirements.txt
```

## Prepare the Physionet Challenge 2012 dataset
```
#create a folder for storing the raw dataset and download the dataset
mkdir /path/to/data/
cd /path/to/data/
wget https://physionet.org/static/published-projects/challenge-2012/predicting-mortality-of-icu-patients-the-physionetcomputing-in-cardiology-challenge-2012-1.0.0.zip
unzip predicting-mortality-of-icu-patients-the-physionetcomputing-in-cardiology-challenge-2012-1.0.0.zip
mv predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/ data/
```

## Run the following script to train one state-of-the-art time series imputation model, SAITS [[1]](#1), without test time adaptation
```
python EHR_imputation/main.py --missing_ratio 0.6 --model_type saits --dataset physionet --input /path/to/data/ --output /path/to/log/ --log_path /path/to/log/ --do_train
```

in the above command, the argument "missing\_ratio" specifies the portion of the non-missing entries that are used for evaluating the performance of imputation model. Specifically, the learned imputation model takes the 1-missing\_ratio of the non-missing entries as input, impute the remaining missing\_ratio entries and compare them against the ground-truth entries. 



## Run the following script to learn statistical quantile rules over the non-missing entries of the training samples
```
python EHR_imputation/obtain_bounds_in_rules.py --dataset physionet --input /path/to/data/ --output /path/to/log/ --log_path /path/to/log/
```

## Run the following script to perform test-time adaptation:
```
python EHR_imputation/main.py --tta_method $method --missing_ratio 0.6 --model_type saits --dataset physionet --input /path/to/data/ --output /path/to/log/ --log_path /path/to/log/
```

in which $method could be either our method or other baseline methods. The choice of $method is as follows:

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
