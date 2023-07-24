# Code of SQRL for object detection task
## Configurations
We prepared a docker file in the folder "docker_files/". To use it, we first of all copy the docker file for the object detection task to the root directory of this project (assuming it is "/path/to/proj/"):
```
cd /path/to/proj/
cp docker_files/Dockerfile_object_detection Dockerfile
```

Then we initialize a docker container for reproducing our experimental results with the following command:
```
docker build -t object_detection .
```

Then within this container, we use the following command to initialize the cuda configurations:
```
apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub && apt-get update && apt-get -y install cuda
```

There will be some error messages when the above command is executed. But it won't influenece the following experiments. 

In the end, the following command is needed to install extra python libraries:
```
cd object_detection/
pip install -r requirements.txt
```

Then following the instructions of the [EfficientPS](https://github.com/DeepSceneSeg/EfficientPS) project to install EfficientPS library:
```
cd efficientNet/
python setup.py develop
cd ../
python setup.py develop
```

## Prepare and Preprocess datasets
### Prepare Kitti dataset
We used the version of the Kitti dataset from [KITTI Panoptic Segmentation Dataset](http://panoptic.cs.uni-freiburg.de/). After downloading this dataset, follow the instructions of the [EfficientPS](https://github.com/DeepSceneSeg/EfficientPS) project to organize the Kitti dataset. 

### Prepare CityScape dataset
Follow the instructions of the [CityScape](https://github.com/DeepSceneSeg/EfficientPS) project to organize the CityScape dataset
* Download gtFine.zip and leftImg8bit_trainvaltest.zip frin CityScape by using the [cityscapesScripts](https://github.com/mcordts/cityscapesScripts)
```
git clone https://github.com/mcordts/cityscapesScripts
cd cityscapesScripts/cityscapesscripts/download/
python downloader.py gtFine_trainvaltest.zip -d /path/to/cityscape/
python downloader.py leftImg8bit_trainvaltest.zip -d /path/to/cityscape/
cd /path/to/cityscape/
unzip gtFine_trainvaltest.zip
unzip leftImg8bit_trainvaltest.zip
```
* By following the instruction of the [CityScape](https://github.com/DeepSceneSeg/EfficientPS) to pre-process the downloaded CityScape dataset:
```
python tools/convert_cityscapes.py /path/to/cityscape/ /path/to/cityscape/
cd /path/to/cityscapesScripts/cityscapesscripts/preparation
python createPanopticImgs.py --dataset-folder /path/to/cityscape/gtFine/ --output-folder /path/to/proj/object_detection/data/cityscapes --set-names val
```

in which we assume that this root directory of this git repo is "/path/to/proj/"

### Prepare CityScape-rainy dataset
We follow the instructions from this [website](https://team.inria.fr/rits/computer-vision/weather-augment/). First, we download the rain files for the validation split of the CityScape from this [link](https://www.rocq.inria.fr/rits_files/download.php?file=computer-vision/weather-augment/weather_cityscapes_leftImg8bit_val_rain_diff.zip)

After unzipping it to "/path/to/cityscape_rainy/weather_datasets/", we download the "weather_dev_toolkit.zip" from the same website (with this [link](https://www.rocq.inria.fr/rits_files/download.php?file=computer-vision/weather-augment/weather_dev_toolkit.zip)), unzip it and use the following command to generate the rainy version of the CityScape dataset:
```
python weather_generate.py cityscapes --cityscapes_root /path/to/citiscape/ --weather rain --output_dir /path/to/cityscape_rainy/ --sequence leftImg8bit/val
```
This step will produce the rainy version of the CityScape dataset, which will be stored in /path/to/cityscape_rainy/. Then we also need to pre-process this dataset in a similar mannar to CityScape dataset:
```
python tools/convert_cityscapes.py /path/to/cityscape_rainy/weather_datasets/weather_cityscapes/leftImg8bit/val/rain/200mm/ /path/to/cityscape_rainy/ --specified_img_dir /path/to/cityscape_rainy/weather_datasets/weather_cityscapes/leftImg8bit/val/rain/200mm/rainy_image/ --specified_gt_dir /path/to/cityscape/gtFine/val/
cd /path/to/cityscapesScripts/cityscapesscripts/preparation
python createPanopticImgs.py --dataset-folder /path/to/cityscape/gtFine/ --output-folder /path/to/cityscape_rainy/ --set-names val
```

in which we take as the input the CityScape images with heaviest rains (200mm) and the real ground-truth labels from "/path/to/cityscape/gtFine/val/".

In the end, we create virtual link to the above data folders:
```
cd /path/to/proj/object_detection/
mkdir data/
ln -s  /path/to/cityscape/ data/cityscapes
ln -s  /path/to/cityscape_rainy/ data/cityscapes_rainy
```



## Download pretrained Efficient model from [here](https://github.com/DeepSceneSeg/EfficientPS#pre-trained-models)
Note that to reproduce our results, we use the model pretrained on Kitti dataset and suppose the pretrained model is stored in "/path/to/model/"


## Learn statistical rules with the following command over the training set of Kitti dataset:
Running
```
python ../rule_learning/rule_learning.py
```
will generate the rules for the semantic analysis task in the JSON format as specified by the file `schema.json`.

## Perform test adaptation:
```
python tools/test_adaptation.py $config_file_name --work_dir /path/to/output/ --checkpoint /path/to/model/model.pth --validate --seed 0 --rule_path /path/to/rule/transformed_bounds.json --tta_method ${method}
```

in which, $config_file_name could be the following choices:

$config_file_name="configs/efficientPS_singlegpu_sample_kitti2.py" for Kitti dataset

$config_file_name="configs/efficientPS_singlegpu_sample_cityscape2.py" for CityScape dataset

$config_file_name="configs/efficientPS_singlegpu_sample_cityscape_rainy2.py" for CityScape-rainy dataset,

and $method could be either our method or other baseline methods. The choice of $method is as follows:

$method=rule: our method

$method=cpl: conjugate pseudo-label [[2]](#2)

$method=rpl: Robust Pseudo-label [[3]](#3)

$method=tent: Tent [[4]](#4)

$method=norm: Batch normalization [[5]](#5)

The execution of the above command can produce one model at each epoch, named as "epoch_x.pth" (x=1,2,...) for the epoch x. Those models are all cached in the folder "/path/to/output/".

## Evaluate the performance of the model "epoch_x.pth" with the following command:
```
python tools/test.py $config_file_name /path/to/output/epoch_x.pth --eval bbox --out_dir /path/to/output/
```



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




