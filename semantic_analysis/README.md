# Code of SQRL for Semantic Analysis task
## Configurations
We prepared a docker file in the folder "docker_files/". To use it, we first of all copy the docker file for the semantic analysis task to the root directory of this project (assuming it is "/path/to/proj/"):
```
cd /path/to/proj/
cp docker_files/Dockerfile_semantic_analysis Dockerfile
```

Then we initialize a docker container for reproducing our experimental results with the following command:
```
docker build -t semantic_analysis .
```

Then within this container, we use the following command to initialize the cuda configurations:
```
apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub && apt-get update && apt-get -y install cuda
```

There will be some error messages when the above command is executed. But it won't influenece the following experiments. 

In the end, the following command is needed to install extra python libraries:
```
cd semantic_analysis/
pip install -r requirements.txt
```

## Prepare financial PhraseBank dataset as suggested by [finBERT](https://github.com/ProsusAI/finBERT):
* Download the financial PhraseBank dataset and pre-process it by following the instructions provided by [finBERT](https://github.com/ProsusAI/finBERT)
* The above step can produce train-valid-test split of the financial PhraseBank dataset, i.e., train.csv, validation.csv, test.csv. Suppose these csv files are stored in the folder "/path/to/data/"


## Prepare pretrained finBERT model as suggested by [finBERT](https://github.com/ProsusAI/finBERT):
```
mkdir /path/to/model/
mkdir /path/to/model/language_model/
mkdir /path/to/model/language_model/finbertTRC2/
mkdir /path/to/model/classifier_model/
mkdir /path/to/model/classifier_model/finbert-sentiment/
wget https://prosus-public.s3-eu-west-1.amazonaws.com/finbert/language-model/pytorch_model.bin -P /path/to/model/language_model/finbertTRC2/
wget https://prosus-public.s3-eu-west-1.amazonaws.com/finbert/finbert-sentiment/pytorch_model.bin -P /path/to/model/classifier_model/finbert-sentiment/
```
### Copy the model configuration file downloaded from [finBERT](https://github.com/ProsusAI/finBERT)  to the above model directory:
```
cp config.json /path/to/model/language_model/finbertTRC2/
cp config.json /path/to/model/classifier_model/finbert-sentiment/
```

## Extract features from pretrained topic models and emotion model
```
python extract_features.py --data_path /path/to/data/ --model_path /path/to/model/`
```

## Learn statistical rules over extracted features
Running
```
python ../rule_learning/rule_learning.py
```
will generate the rules for the semantic analysis task in the CSV format as specified by the file `schema.json`.
In order to also get the rules generated over the validation set, first set the `schema_file` variable in `../rule_learning/rule_learning.py` to `schema_valid.json` and then run the script again.

## Perform test-time adaptation over the pretrained finBERT model:
```
python test_time_adaptation/tta.py --tta_method $method --data_path /path/to/data/ --model_path /path/to/model/ --epochs 50 --lr 1e-3
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

