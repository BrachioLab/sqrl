# Code of SQRL for image classification task

## Configurations
We prepared a docker file in the folder "docker_files/". To use it, we first of all copy the docker file for the image classification task to the root directory of this project (assuming it is "/path/to/proj/"):
```
cd /path/to/proj/
cp docker_files/Dockerfile_image_classification Dockerfile
```

Then we initialize a docker container for reproducing our experimental results with the following command:
```
docker build -t image_classification .
```

Then within this container, we use the following command to initialize the cuda configurations:
```
apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub && apt-get update && apt-get -y install cuda
```

There will be some error messages when the above command is executed. But it won't influenece the following experiments. 

In the end, the following command is needed to install extra python libraries:
```
cd image_classification/
pip install -r requirements.txt
```

## Preparing data by downloading the full ImageNet dataset with the following kaggle command:
```
kaggle competitions download -c imagenet-object-localization-challenge
```
## Pre-process the ImageNet dataset:
Note that the validation and test splits of the imagenet dataset are not organized in a way in which each folder represents a class. We can use the existing script from other project, such as the one from [here](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) to reorganize the imagenet dataset. After this step, suppose the imagenet dataset is stored in a folder "/path/to/data" which takes "train", "val" and "test" as its subfolder.

## Train a ResNet-34 model without test-time adaptation over the training set of the ImageNet dataset, but only using the 17 metaclasses defined in ImageNet-X dataset
```
python train_and_test_time_adaptation.py --train --work_dir /path/to/output/ --data_dir /path/to/data/ --batch_size 256 --full_model --lr 0.1
```
which ends up with a ResNet-34 model cached in "/path/to/output/". The hyper-parameters such as "batch_size" and "lr" could be adjusted. Other hyper-parameters include "--epochs".


We prepared the rules that we generated for the experiments, which are stored in "image_classification/rule_after_validations/". You can just copy those rules to the directory "/path/to/output/train/" so that you can skip the following two steps for learning and validating statistical rules, i.e.:
```
cp rule_after_validations/* /path/to/output/train/
```

Otherwise, you can go through the following two steps to create rules from scratch:

## Learn statistical rules from ImageNet-X dataset
```
TO BE DONE
```
The code for the rule learning component of the SQRL framework is being finalized and will be released soon.
For now, we provide the learned rules for the image classification task in the folder `rule_after_validations/`.

## Validating rules by removing rules that are not generalized well from training set to validation set
```
python train_and_test_time_adaptation.py --validate --work_dir /path/to/output/ --data_dir /path/to/data/ --batch_size 256 --full_model --lr 0.1 --topk 200 --cache_dir /path/to/output/train/ --load_train_val_data
```

in which "--topk" specifies the number of rules selected within each meta-class, "--cache_dir" stores the data and model obtained in the prior training process and "--load_train_val_data" means loading the cached data from the cache_dir.

## Perform test-time adaptations on the model trained without test-time adaptation.
```
python train_and_test_time_adaptation.py --tta_method ${method} --load_train_val_data --load_test_data --load_filtered_rules --cache_dir /path/to/output/train/ --work_dir /path/to/output_test_time/ --data_dir /path/to/data/ --batch_size ${batch_size} --lr ${lr} --seed ${seed} --epochs ${eps}
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
