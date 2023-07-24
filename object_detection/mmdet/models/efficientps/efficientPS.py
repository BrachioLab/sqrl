from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

import geffnet

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, multiclass_nms)
from .. import builder
from ..registry import EFFICIENTPS
from .base import BaseDetector
from mmdet.ops.norm import norm_cfg
from mmdet.ops.roi_sampling import roi_sampling, invert_roi_bbx

import time
import cv2
import numpy as np

import json

from .statistical_rule_loss import *
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from baseline_methods import conjugate_pl, robust_pl

LAMBDA = 0.001
classes = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


# attribute_name_ls = ["aspect_ratios", "xc", "yb", "widths", "heights", "areas"]
attribute_name_ls = ["aspect_ratios", "xc", "widths", "heights", "areas"]
# aspect_ratios = {"bicycle": [0.185185179, 12.5], "traffic sign": [0.0313588865, 27.0], "traffic light": [0.0792602375, 9.0], "car": [0.0507614203, 20.0], "person": [0.318181813, 23.2000008], "rider": [0.555555582, 5.76000023], "motorcycle": [0.141666666, 5.21428585], "bus": [0.173515975, 8.29166698], "truck": [0.212017581, 6.41891909], "caravan": [0.456521749, 2.36543202], "trailer": [0.283950627, 3.28512406]}

# widths = {"bicycle": [2.0, 677.0], "traffic sign": [2.0, 1590.0], "traffic light": [2.0, 2047.0], "car": [3.0, 1147.0], "person": [2.0, 503.0], "rider": [5.0, 400.0], "motorcycle": [3.0, 834.0], "bus": [8.0, 1423.0], "truck": [12.0, 2047.0], "caravan": [35.0, 623.0], "trailer": [31.0, 369.0]}

# heights = {"bicycle": [5.0, 543.0], "traffic sign": [2.0, 626.0], "traffic light": [4.0, 412.0], "car": [3.0, 1024.0], "person": [5.0, 914.0], "rider": [5.0, 639.0], "motorcycle": [4.0, 501.0], "bus": [6.0, 861.0], "truck": [11.0, 895.0], "caravan": [21.0, 958.0], "trailer": [18.0, 795.0]}

# areas = {"bicycle": [[[0, 57828.6], [4.47213595499958, 1063.6245813255728]], [[57828.6, 115641.2], [60.47520152922188, 941.2625829172218]], [[115641.2, 173453.8], [248.91765706755317, 924.9963513441553]], [[173453.8, 231266.4], [126.30122723077555, 869.6599910309776]], [[231266.4, 2097152], [113.58917201916739, 556.7463515821186]]], "traffic sign": [[[0, 124197.6], [10.404326023342406, 1124.3598178519187]], [[124197.6, 248383.2], [345.52930411182206, 898.5683056952321]], [[248383.2, 372568.80000000005], [354.46085820581095, 354.46085820581095]], [[372568.80000000005, 496754.4], [367.4479554984624, 367.4479554984624]], [[496754.4, 2097152], [393.80102843949, 393.80102843949]]], "traffic light": [[[0, 168682.4], [31.184932259025352, 1104.3922536852565]], [[168682.4, 337352.8], [414.86443568953945, 524.9695229249028]], [[337352.8, 506023.19999999995], []], [[506023.19999999995, 674693.6], []], [[674693.6, 2097152], [306.0004084964594, 306.0004084964594]]], "car": [[[0, 145398.6], [6.5, 1045.038396423787]], [[145398.6, 290782.2], [9.433981132056603, 927.4034990229442]], [[290782.2, 436165.80000000005], [109.65856099730654, 847.3678363025116]], [[436165.80000000005, 581549.4], [109.32977636490436, 774.785938179056]], [[581549.4, 2097152], [57.019733426244635, 740.0]]], "person": [[[0, 90156.8], [3.2015621187164243, 1100.763939271268]], [[90156.8, 180289.6], [8.06225774829855, 938.0767559213905]], [[180289.6, 270422.4], [35.21718330588067, 905.6026998634666]], [[270422.4, 360555.2], [276.02400257948585, 832.5998438625844]], [[360555.2, 2097152], [104.59684507670391, 351.7786946362727]]], "rider": [[[0, 45315.6], [32.26840560052511, 1019.6147556798106]], [[45315.6, 90586.2], [43.83206588788623, 953.368370568271]], [[90586.2, 135856.8], [153.43158084305853, 867.0784278253035]], [[135856.8, 181127.4], [64.66065264130884, 907.0551251164396]], [[181127.4, 2097152], [89.04493247793498, 873.105091040019]]], "motorcycle": [[[0, 44716.8], [14.7648230602334, 1013.6547735792498]], [[44716.8, 89415.6], [69.35776813018136, 975.2497116123644]], [[89415.6, 134114.40000000002], [57.56083738098326, 903.0566150579929]], [[134114.40000000002, 178813.2], [33.094561486745825, 866.3420225292087]], [[178813.2, 2097152], [598.0974000946669, 808.5001546072826]]], "bus": [[[0, 135048.0], [38.30469945058961, 1021.4029567217827]], [[135048.0, 270042.0], [139.89013546351293, 879.8377691370154]], [[270042.0, 405036.0], [567.0282179927203, 773.9266761134417]], [[405036.0, 540030.0], [289.8309852310481, 690.0371366817876]], [[540030.0, 2097152], [514.7632951172801, 637.2332775365705]]], "truck": [[[0, 177785.2], [41.6293165929973, 1052.5549866871563]], [[177785.2, 355438.4], [167.20047846821492, 900.9252188722436]], [[355438.4, 533091.6000000001], [550.4682097996214, 787.9961928842042]], [[533091.6000000001, 710744.8], [476.94994496278116, 714.915554453811]], [[710744.8, 2097152], [22.005681084665387, 417.94048619390776]]], "caravan": [[[0, 92727.6], [22.588713996153036, 980.060457318833]], [[92727.6, 184489.2], [265.7277742352124, 862.3770926920542]], [[184489.2, 276250.80000000005], [205.5024330756208, 553.9072575801837]], [[276250.80000000005, 368012.4], []], [[368012.4, 2097152], [725.7280826866216, 821.1633515933355]]], "trailer": [[[0, 39640.8], [73.824115301167, 1013.9483468106253]], [[39640.8, 78597.6], [373.71011492867035, 936.8608487923914]], [[78597.6, 117554.40000000001], [386.87465670420954, 874.6913741429031]], [[117554.40000000001, 156511.2], [420.1743685661942, 807.3755322029521]], [[156511.2, 2097152], [862.1915390445444, 902.0034645166281]]]}



# aspect_ratios = {"bicycle": [4.5, 5.254545497999996], "traffic sign": [4.229316198899984, 4.66666651], "traffic light": [3.7708332525, 4.0], "car": [2.6400001, 3.0301986309449975], "person": [4.83333349, 5.174673179224015], "rider": [3.294553641299998, 3.763395216599992], "motorcycle": [2.4604303112800006, 4.0584051048], "bus": [2.0116175298999996, 6.6534937065999955], "truck": [2.27272725, 4.0869565], "caravan": [1.4193548, 2.36543202], "trailer": [1.384176555, 3.28512406]}
# widths = {"bicycle": [318.3959999999997, 390.99999999999454], "traffic sign": [163.0, 179.0], "traffic light": [98.0, 118.0], "car": [542.0, 564.0], "person": [202.0, 227.2400000000016], "rider": [207.32669999999993, 288.0], "motorcycle": [339.0, 500.34000000000003], "bus": [720.7655000000001, 1329.8499999999997], "truck": [583.0, 1304.0], "caravan": [478.59999999999997, 623.0], "trailer": [312.25, 369.0]}
# heights = {"bicycle": [311.994, 361.0], "traffic sign": [164.0, 188.0], "traffic light": [154.0, 166.75], "car": [528.5499999999993, 568.5544999999994], "person": [475.2376000000016, 529.0], "rider": [395.0, 491.0], "motorcycle": [341.1, 450.2600000000001], "bus": [481.95, 778.3799999999998], "truck": [592.4484000000012, 877.0], "caravan": [408.79999999999984, 958.0], "trailer": [333.25, 795.0]}
# areas = {"bicycle": [[[0, 57828.6], [1001.2445331796233, 1012.7350344487941]], [[57828.6, 115641.2], [916.9963195127884, 941.2625829172218]], [[115641.2, 173453.8], [804.0613850109147, 924.9963513441553]], [[173453.8, 231266.4], [443.18901903237406, 869.6599910309776]], [[231266.4, 2097152], [113.58917201916739, 556.7463515821186]]], "traffic sign": [[[0, 124197.6], [1034.527433110185, 1044.311923069817]], [[124197.6, 248383.2], [345.52930411182206, 898.5683056952321]], [[248383.2, 372568.80000000005], [354.46085820581095, 354.46085820581095]], [[372568.80000000005, 496754.4], [367.4479554984624, 367.4479554984624]], [[496754.4, 2097152], [393.80102843949, 393.80102843949]]], "traffic light": [[[0, 168682.4], [1026.2883805596114, 1042.7435955075873]], [[168682.4, 337352.8], [414.86443568953945, 524.9695229249028]], [[337352.8, 506023.19999999995], []], [[506023.19999999995, 674693.6], []], [[674693.6, 2097152], [306.0004084964594, 306.0004084964594]]], "car": [[[0, 145398.6], [990.0907601631914, 998.7345001296898]], [[145398.6, 290782.2], [888.4438624472482, 912.0901819447461]], [[290782.2, 436165.80000000005], [818.9058580601411, 847.3678363025116]], [[436165.80000000005, 581549.4], [713.3184746861137, 774.785938179056]], [[581549.4, 2097152], [614.4871845693773, 740.0]]], "person": [[[0, 90156.8], [1000.0975600363566, 1007.2859942087355]], [[90156.8, 180289.6], [905.7396239093757, 938.0767559213905]], [[180289.6, 270422.4], [793.86286599135, 905.6026998634666]], [[270422.4, 360555.2], [486.7206322892076, 832.5998438625844]], [[360555.2, 2097152], [148.34337391323376, 351.7786946362727]]], "rider": [[[0, 45315.6], [981.50283882535, 1003.2328991814413]], [[45315.6, 90586.2], [934.8305372802884, 953.368370568271]], [[90586.2, 135856.8], [538.5020891324378, 867.0784278253035]], [[135856.8, 181127.4], [218.3565487528458, 907.0551251164396]], [[181127.4, 2097152], [89.04493247793498, 873.105091040019]]], "motorcycle": [[[0, 44716.8], [983.5285754542721, 1011.5238504355693]], [[44716.8, 89415.6], [903.3057345107469, 975.2497116123644]], [[89415.6, 134114.40000000002], [730.0922190727474, 903.0566150579929]], [[134114.40000000002, 178813.2], [827.2843485703797, 866.3420225292087]], [[178813.2, 2097152], [614.9879686283019, 808.5001546072826]]], "bus": [[[0, 135048.0], [928.2474305578689, 1013.3122148698186]], [[135048.0, 270042.0], [733.3595639248185, 879.8377691370154]], [[270042.0, 405036.0], [698.6530472364526, 773.9266761134417]], [[405036.0, 540030.0], [338.8469418483809, 690.0371366817876]], [[540030.0, 2097152], [514.7632951172801, 637.2332775365705]]], "truck": [[[0, 177785.2], [958.5549518232432, 997.9400032066056]], [[177785.2, 355438.4], [832.6955766357102, 900.9252188722436]], [[355438.4, 533091.6000000001], [738.7402652670479, 787.9961928842042]], [[533091.6000000001, 710744.8], [476.94994496278116, 714.915554453811]], [[710744.8, 2097152], [22.005681084665387, 417.94048619390776]]], "caravan": [[[0, 92727.6], [649.667615261747, 980.060457318833]], [[92727.6, 184489.2], [434.68062988819736, 862.3770926920542]], [[184489.2, 276250.80000000005], [205.5024330756208, 553.9072575801837]], [[276250.80000000005, 368012.4], []], [[368012.4, 2097152], [771.1596230221551, 821.1633515933355]]], "trailer": [[[0, 39640.8], [793.6489430123436, 1013.9483468106253]], [[39640.8, 78597.6], [577.105720388865, 936.8608487923914]], [[78597.6, 117554.40000000001], [661.7792663063524, 874.6913741429031]], [[117554.40000000001, 156511.2], [420.1743685661942, 807.3755322029521]], [[156511.2, 2097152], [862.1915390445444, 902.0034645166281]]]}


aspect_ratios = {"bicycle": [0.354838699, 5.5],
                "traffic sign": [0.181818187, 5.23809528],
                "traffic light": [0.540540516, 4.33333349],
                "car": [0.263157904, 3.67010307],
                "person": [0.735294104, 5.73333311],
                "rider": [0.678571403, 3.83999991],
                "motorcycle": [0.239999995, 4.0625],
                "bus": [0.173515975, 8.29166698],
                "truck": [0.251184821, 4.0869565],
                "caravan": [0.456521749, 2.36543202],
                "trailer": [0.283950627, 3.28512406]}

widths = {"bicycle": [4.0, 430.0],
          "traffic sign": [5.0, 206.0],
          "traffic light": [4.0, 139.0],
          "car": [12.0, 603.0],
          "person": [6.0, 251.0],
          "rider": [7.0, 302.0],
          "motorcycle": [5.0, 503.0],
          "bus": [8.0, 1423.0],
          "truck": [13.0, 1304.0],
          "caravan": [35.0, 623.0],
          "trailer": [31.0, 369.0]}

heights = {"bicycle": [11.0, 372.0], 
           "traffic sign": [5.0, 224.0], 
           "traffic light": [7.0, 179.0], 
           "car": [10.0, 627.0],
           "person": [12.0, 576.0], 
           "rider": [11.0, 502.0], 
           "motorcycle": [6.0, 459.0], 
           "bus": [6.0, 861.0],
           "truck": [15.0, 877.0], 
           "caravan": [21.0, 958.0], 
           "trailer": [18.0, 795.0]}
           
areas = {"bicycle": [[[0, 57828.6], [34.23448553724738, 1014.3205854166621]], [[57828.6, 115641.2], [60.47520152922188, 941.2625829172218]], [[115641.2, 173453.8], [248.91765706755317, 924.9963513441553]], [[173453.8, 231266.4], [126.30122723077555, 869.6599910309776]], [[231266.4, 2097152], [113.58917201916739, 556.7463515821186]]],
        "traffic sign": [[[0, 124197.6], [81.7572626743337, 1058.538260999573]], [[124197.6, 248383.2], [345.52930411182206, 898.5683056952321]], [[248383.2, 372568.80000000005], [354.46085820581095, 354.46085820581095]], [[372568.80000000005, 496754.4], [367.4479554984624, 367.4479554984624]], [[496754.4, 2097152], [393.80102843949, 393.80102843949]]],
        "traffic light": [[[0, 168682.4], [99.02019995940222, 1053.4914570132973]], [[168682.4, 337352.8], [414.86443568953945, 524.9695229249028]], [[337352.8, 506023.19999999995], []], [[506023.19999999995, 674693.6], []], [[674693.6, 2097152], [306.0004084964594, 306.0004084964594]]],
        "car": [[[0, 145398.6], [38.6846222677694, 1007.7500930290207]], [[145398.6, 290782.2], [21.360009363293827, 912.0901819447461]], [[290782.2, 436165.80000000005], [109.65856099730654, 847.3678363025116]], [[436165.80000000005, 581549.4], [109.32977636490436, 774.785938179056]], [[581549.4, 2097152], [57.019733426244635, 740.0]]],
        "person": [[[0, 90156.8], [49.24428900898052, 1013.5543892658154]], [[90156.8, 180289.6], [8.06225774829855, 938.0767559213905]], [[180289.6, 270422.4], [35.21718330588067, 905.6026998634666]], [[270422.4, 360555.2], [276.02400257948585, 832.5998438625844]], [[360555.2, 2097152], [104.59684507670391, 351.7786946362727]]],
        "rider": [[[0, 45315.6], [45.254833995939045, 1004.4082835182115]], [[45315.6, 90586.2], [43.83206588788623, 953.368370568271]], [[90586.2, 135856.8], [153.43158084305853, 867.0784278253035]], [[135856.8, 181127.4], [64.66065264130884, 907.0551251164396]], [[181127.4, 2097152], [89.04493247793498, 873.105091040019]]],
        "motorcycle": [[[0, 44716.8], [19.704060495238032, 1011.5238504355693]], [[44716.8, 89415.6], [69.35776813018136, 975.2497116123644]], [[89415.6, 134114.40000000002], [57.56083738098326, 903.0566150579929]], [[134114.40000000002, 178813.2], [33.094561486745825, 866.3420225292087]], [[178813.2, 2097152], [598.0974000946669, 808.5001546072826]]],
        "bus": [[[0, 135048.0], [38.30469945058961, 1021.4029567217827]], [[135048.0, 270042.0], [139.89013546351293, 879.8377691370154]], [[270042.0, 405036.0], [567.0282179927203, 773.9266761134417]], [[405036.0, 540030.0], [289.8309852310481, 690.0371366817876]], [[540030.0, 2097152], [514.7632951172801, 637.2332775365705]]],
        "truck": [[[0, 177785.2], [50.224496015390734, 997.9400032066056]], [[177785.2, 355438.4], [167.20047846821492, 900.9252188722436]], [[355438.4, 533091.6000000001], [550.4682097996214, 787.9961928842042]], [[533091.6000000001, 710744.8], [476.94994496278116, 714.915554453811]], [[710744.8, 2097152], [22.005681084665387, 417.94048619390776]]],
        "caravan": [[[0, 92727.6], [22.588713996153036, 980.060457318833]], [[92727.6, 184489.2], [265.7277742352124, 862.3770926920542]], [[184489.2, 276250.80000000005], [205.5024330756208, 553.9072575801837]], [[276250.80000000005, 368012.4], []], [[368012.4, 2097152], [725.7280826866216, 821.1633515933355]]],
        "trailer": [[[0, 39640.8], [73.824115301167, 1013.9483468106253]], [[39640.8, 78597.6], [373.71011492867035, 936.8608487923914]], [[78597.6, 117554.40000000001], [386.87465670420954, 874.6913741429031]], [[117554.40000000001, 156511.2], [420.1743685661942, 807.3755322029521]], [[156511.2, 2097152], [862.1915390445444, 902.0034645166281]]]}


# aspect_ratio_bounds = [
#     (0.735294104, 5.73333311),
#     (0.678571403, 3.83999991),
#     (0.263157904, 3.67010307),
#     (0.251184821, 4.0869565),
#     (0.173515975, 8.29166698),
#     (),
#     (0.239999995, 4.0625),
#     (0.354838699, 5.5)
# ]

# aspect_ratios = {"bicycle": [0.354838699, 5.5],
#                 "traffic sign": [0.181818187, 5.23809528],
#                 "traffic light": [0.540540516, 4.33333349],
#                 "car": [0.263157904, 3.67010307],
#                 "person": [0.735294104, 5.73333311],
#                 "rider": [0.678571403, 3.83999991],
#                 "motorcycle": [0.239999995, 4.0625],
#                 "bus": [0.173515975, 8.29166698],
#                 "truck": [0.251184821, 4.0869565],
#                 "caravan": [0.456521749, 2.36543202],
#                 "trailer": [0.283950627, 3.28512406]}

# widths = {"bicycle": [4.0, 430.0],
#           "traffic sign": [5.0, 206.0],
#           "traffic light": [4.0, 139.0],
#           "car": [12.0, 603.0],
#           "person": [6.0, 251.0],
#           "rider": [7.0, 302.0],
#           "motorcycle": [5.0, 503.0],
#           "bus": [8.0, 1423.0],
#           "truck": [13.0, 1304.0],
#           "caravan": [35.0, 623.0],
#           "trailer": [31.0, 369.0]}

# heights = {"bicycle": [11.0, 372.0], 
#            "traffic sign": [5.0, 224.0], 
#            "traffic light": [7.0, 179.0], 
#            "car": [10.0, 627.0],
#            "person": [12.0, 576.0], 
#            "rider": [11.0, 502.0], 
#            "motorcycle": [6.0, 459.0], 
#            "bus": [6.0, 861.0],
#            "truck": [15.0, 877.0], 
#            "caravan": [21.0, 958.0], 
#            "trailer": [18.0, 795.0]}
           
# areas = {"bicycle": [[[0, 57828.6], [34.23448553724738, 1014.3205854166621]], [[57828.6, 115641.2], [60.47520152922188, 941.2625829172218]], [[115641.2, 173453.8], [248.91765706755317, 924.9963513441553]], [[173453.8, 231266.4], [126.30122723077555, 869.6599910309776]], [[231266.4, 2097152], [113.58917201916739, 556.7463515821186]]],
#         "traffic sign": [[[0, 124197.6], [81.7572626743337, 1058.538260999573]], [[124197.6, 248383.2], [345.52930411182206, 898.5683056952321]], [[248383.2, 372568.80000000005], [354.46085820581095, 354.46085820581095]], [[372568.80000000005, 496754.4], [367.4479554984624, 367.4479554984624]], [[496754.4, 2097152], [393.80102843949, 393.80102843949]]],
#         "traffic light": [[[0, 168682.4], [99.02019995940222, 1053.4914570132973]], [[168682.4, 337352.8], [414.86443568953945, 524.9695229249028]], [[337352.8, 506023.19999999995], []], [[506023.19999999995, 674693.6], []], [[674693.6, 2097152], [306.0004084964594, 306.0004084964594]]],
#         "car": [[[0, 145398.6], [38.6846222677694, 1007.7500930290207]], [[145398.6, 290782.2], [21.360009363293827, 912.0901819447461]], [[290782.2, 436165.80000000005], [109.65856099730654, 847.3678363025116]], [[436165.80000000005, 581549.4], [109.32977636490436, 774.785938179056]], [[581549.4, 2097152], [57.019733426244635, 740.0]]],
#         "person": [[[0, 90156.8], [49.24428900898052, 1013.5543892658154]], [[90156.8, 180289.6], [8.06225774829855, 938.0767559213905]], [[180289.6, 270422.4], [35.21718330588067, 905.6026998634666]], [[270422.4, 360555.2], [276.02400257948585, 832.5998438625844]], [[360555.2, 2097152], [104.59684507670391, 351.7786946362727]]],
#         "rider": [[[0, 45315.6], [45.254833995939045, 1004.4082835182115]], [[45315.6, 90586.2], [43.83206588788623, 953.368370568271]], [[90586.2, 135856.8], [153.43158084305853, 867.0784278253035]], [[135856.8, 181127.4], [64.66065264130884, 907.0551251164396]], [[181127.4, 2097152], [89.04493247793498, 873.105091040019]]],
#         "motorcycle": [[[0, 44716.8], [19.704060495238032, 1011.5238504355693]], [[44716.8, 89415.6], [69.35776813018136, 975.2497116123644]], [[89415.6, 134114.40000000002], [57.56083738098326, 903.0566150579929]], [[134114.40000000002, 178813.2], [33.094561486745825, 866.3420225292087]], [[178813.2, 2097152], [598.0974000946669, 808.5001546072826]]],
#         "bus": [[[0, 135048.0], [38.30469945058961, 1021.4029567217827]], [[135048.0, 270042.0], [139.89013546351293, 879.8377691370154]], [[270042.0, 405036.0], [567.0282179927203, 773.9266761134417]], [[405036.0, 540030.0], [289.8309852310481, 690.0371366817876]], [[540030.0, 2097152], [514.7632951172801, 637.2332775365705]]],
#         "truck": [[[0, 177785.2], [50.224496015390734, 997.9400032066056]], [[177785.2, 355438.4], [167.20047846821492, 900.9252188722436]], [[355438.4, 533091.6000000001], [550.4682097996214, 787.9961928842042]], [[533091.6000000001, 710744.8], [476.94994496278116, 714.915554453811]], [[710744.8, 2097152], [22.005681084665387, 417.94048619390776]]],
#         "caravan": [[[0, 92727.6], [22.588713996153036, 980.060457318833]], [[92727.6, 184489.2], [265.7277742352124, 862.3770926920542]], [[184489.2, 276250.80000000005], [205.5024330756208, 553.9072575801837]], [[276250.80000000005, 368012.4], []], [[368012.4, 2097152], [725.7280826866216, 821.1633515933355]]],
#         "trailer": [[[0, 39640.8], [73.824115301167, 1013.9483468106253]], [[39640.8, 78597.6], [373.71011492867035, 936.8608487923914]], [[78597.6, 117554.40000000001], [386.87465670420954, 874.6913741429031]], [[117554.40000000001, 156511.2], [420.1743685661942, 807.3755322029521]], [[156511.2, 2097152], [862.1915390445444, 902.0034645166281]]]}

def get_stat_bounds(stat: dict, cls: int):
    cls_str = classes[cls]
    return stat[cls_str] if cls_str in stat else ()

def get_area_bounds(cls: int, a: float):
    cls_str = classes[cls]
    area_buckets = areas[cls_str] if cls_str in areas else []
    if area_buckets == []:
        return ()
    for bucket in area_buckets:
        area_bounds = bucket[0]
        if a <= area_bounds[0] or a > area_bounds[1]:
            continue
        return (bucket[1][0], bucket[1][1])
    return ()

def get_by_attr_bounds(a: float, area_buckets: list):
    # area_buckets = areas[cls_str] if cls_str in areas else []
    if area_buckets == []:
        return ()
    for bucket in area_buckets:
        area_bounds = bucket[0]
        if a <= area_bounds[0] or a > area_bounds[1]:
            continue
        return (bucket[1][0], bucket[1][1])
    return ()

def read_rules_from_files(rule_path):
    with open(rule_path) as f:
        json_obj = json.load(f)
    
    rule_bound_mappings_by_class = dict()
    for class_type in json_obj:
        rule_bounds_by_attribute = json_obj[class_type]
        rule_bound_mappings_by_class[class_type] = dict()
        for attr in rule_bounds_by_attribute:
            # if attr not in rule_bound_mappings_by_class:
            #     rule_bound_mappings_by_class[attr] = dict()    
            if not attr.endswith("_groups"):
                if "yb" in attr:
                    continue
                rule_bound_mappings_by_class[class_type][attr] = rule_bounds_by_attribute[attr][0]
            else:
                attr_name = attr.split("_groups")[0]
                for _, group_bound_ls in rule_bounds_by_attribute[attr].items():
                    lb = -np.inf
                    ub = np.inf
                    if "ubound" in group_bound_ls:
                        ub = group_bound_ls["ubound"]
                    if "lbound" in group_bound_ls:
                        lb = group_bound_ls["lbound"]

                    conf_bound_ls = group_bound_ls["conf_bounds"]
                    for by_attr in conf_bound_ls:
                        if "yb" in by_attr:
                            continue
                        
                        local_bound = conf_bound_ls[by_attr][0]
                        group_attr_key = attr_name + "#" + by_attr
                        if group_attr_key not in rule_bound_mappings_by_class[class_type]:
                            rule_bound_mappings_by_class[class_type][group_attr_key] = []
                        rule_bound_mappings_by_class[class_type][group_attr_key].append([[lb, ub], local_bound])

    # rule_bound_mappings_by_class = rule_bound_mappings_by_class


    print("all rules are::")
    
    

    for attr in attribute_name_ls:
        for class_name in rule_bound_mappings_by_class:
            print(attr + "::" + class_name + "::" + str(rule_bound_mappings_by_class[class_name][attr]))
    
    for attr1 in attribute_name_ls:
        for attr2 in attribute_name_ls:
            if not attr1 == attr2:
                for class_name in rule_bound_mappings_by_class:
                    group_attr_key = attr1 + "#" + attr2
                    print(attr2 + " by " + attr1 + "::" + class_name + "::" + str(rule_bound_mappings_by_class[class_name][group_attr_key]))

    print()
    return rule_bound_mappings_by_class


@EFFICIENTPS.register_module
class EfficientPS(BaseDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 semantic_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        assert backbone is not None
        assert rpn_head is not None
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert mask_roi_extractor is not None           
        assert mask_head is not None
        assert semantic_head is not None

        super(EfficientPS, self).__init__()

        self.eff_backbone_flag = False if 'efficient' not in backbone['type'] else True

        if self.eff_backbone_flag == False:
            self.backbone = builder.build_backbone(backbone)
        else:
            self.backbone = geffnet.create_model(backbone['type'], 
                                                 pretrained=True if pretrained is not None else False,
                                                 se=False, 
                                                 act_layer=backbone['act_cfg']['type'],
                                                 norm_layer=norm_cfg[backbone['norm_cfg']['type']][1]) 

        
        self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        self.rpn_head = builder.build_head(rpn_head)

        self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
        self.bbox_head = builder.build_head(bbox_head)

        self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
        self.share_roi_extractor = True
        self.mask_head = builder.build_head(mask_head)

        self.semantic_head = builder.build_head(semantic_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if train_cfg is not None:
            self.rule_path = train_cfg.rule_path
        else:
            self.rule_path = None
        if self.rule_path is not None:
            # self.read_rules_from_files()
            self.rule_bound_mappings_by_class = read_rules_from_files(self.rule_path)
        self.num_classes = semantic_head['num_classes']
        self.num_stuff = self.num_classes - bbox_head['num_classes'] + 1
        self.init_weights(pretrained=pretrained)

    



    def init_weights(self, pretrained=None):
        if self.eff_backbone_flag == False:
            self.backbone.init_weights(pretrained=pretrained)

        self.neck.init_weights()

        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)

        self.rpn_head.init_weights()
        self.bbox_roi_extractor.init_weights()
        self.bbox_head.init_weights()
        self.mask_head.init_weights()
        self.mask_roi_extractor.init_weights()
        self.semantic_head.init_weights() 

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def forward_dummy(self, img): #leave it for now
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(device=img.device)
        # bbox head
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            outs = outs + (mask_pred, )
        return outs

    def assign_result(self, x, proposal_list,
                      img, gt_bboxes, gt_labels, gt_bboxes_ignore):
        bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
        bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
        num_imgs = img.size(0)
        if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
            sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        return sampling_results

    def get_transformed_width(self, bbox, img_shape):
        width = (bbox[2] - bbox[0])/img_shape[1]*2048
        return width

    def get_transformed_bbox(self, bbox, img_shape):
        transformed_ratio = torch.tensor([2048/img_shape[1], 1024/img_shape[0], 2048/img_shape[1], 1024/img_shape[0]])
        transformed_ratio = transformed_ratio.to(bbox.device)
        transformed_bbox = bbox[0:4].view(-1)*transformed_ratio.view(-1)
        return transformed_bbox
    
    def get_transformed_height(self, bbox, img_shape):
        width = (bbox[3] - bbox[1])/img_shape[0]*1024
        return width

    def get_aspect_ratio_violation(self, bbox, cls, img_shape, lam = 0.0):
        height = self.get_transformed_height(bbox, img_shape) 
        width = self.get_transformed_width(bbox, img_shape)
        # width = bbox[2] - bbox[0]
        # height = bbox[3] - bbox[1]
        aspect_ratio = height / (width + 0.001)

        bounds = torch.tensor(get_stat_bounds(aspect_ratios, cls))
        if len(bounds) == 0:
            bounds = torch.tensor((aspect_ratio, aspect_ratio))

        assert len(bounds) == 2

        loss = (aspect_ratio - bounds[0]) * (aspect_ratio - bounds[1])
        loss = torch.clamp(loss, min=0.0, max=1.0)
        # print("Aspect Ratio Loss:", loss, aspect_ratio, bounds)
        # loss = torch.tanh(loss)
        
        return lam * loss

    def get_aspect_ratio_violation2(self, bbox, cls, img_shape, lam = 0.0, epsilon=5):
        # width = bbox[2] - bbox[0]
        # height = bbox[3] - bbox[1]
        height = self.get_transformed_height(bbox, img_shape) 
        width = self.get_transformed_width(bbox, img_shape)
        aspect_ratio = height / (width + 0.001)

        bounds = torch.tensor(get_stat_bounds(aspect_ratios, cls))
        if len(bounds) == 0:
            bounds = torch.tensor((aspect_ratio, aspect_ratio))

        assert len(bounds) == 2

        if bbox[0] > epsilon and bbox[2] < img_shape[1] - epsilon:
            loss = (aspect_ratio - bounds[0]) * (aspect_ratio - bounds[1])
            loss = torch.clamp(loss, min=0.0, max=1.0)
        else:
            loss = (bounds[0] - aspect_ratio)# * (aspect_ratio - bounds[1])
            loss = torch.clamp(loss, min=0.0, max=1.0)
        # print("Aspect Ratio Loss:", loss, aspect_ratio, bounds)
        # loss = torch.tanh(loss)
        
        return lam * loss

    def get_height_violations(self, bbox, cls, img_shape, lam = 0.0):
        height = self.get_transformed_height(bbox, img_shape)
        # height = bbox[3] - bbox[1]
        bounds = torch.tensor(get_stat_bounds(heights, cls))
        if len(bounds) == 0:
            bounds = torch.tensor((height, height))

        assert len(bounds) == 2

        loss = (height - bounds[0]) * (height - bounds[1])
        loss = torch.clamp(loss, min=0.0, max=1.0)

        return lam * loss

    def tta_loss_bbox(self, bbox, losses):
        
        if len(bbox) > 0:
            max_norm = torch.max(bbox**2).item()

            # loss = -0.5*torch.norm(bbox/max_norm, p="fro")
            loss = -0.5*torch.mean(bbox**2/max_norm) + 1

            losses['tent_bbox_loss'] = loss
        else:
            synth_data = torch.rand([1,10], requires_grad=True)

            losses['tent_bbox_loss'] = torch.mean(synth_data)
        # return loss

    def get_statistics_by_name(self, bbox, height, width, attr_name):
        
        if attr_name == "aspect_ratios":
            res = height / (width + 0.001)
        elif attr_name == "xc":
            res = (bbox[0] + bbox[2])/2
        elif attr_name == "yb":
            res = bbox[3]
        elif attr_name == "widths":
            res = width
        elif attr_name == "heights":
            res = height
        elif attr_name == 'areas':
            res = height*width
        return res

    def get_single_attr_violations(self, bbox, cls, attr_name, img_shape, lam = 0.0):
        # width = self.get_transformed_width(bbox, img_shape)
        height = self.get_transformed_height(bbox, img_shape) 
        width = self.get_transformed_width(bbox, img_shape)
        transformed_bbox = self.get_transformed_bbox(bbox, img_shape)
        cls_str = classes[cls]
        if cls_str not in self.rule_bound_mappings_by_class:
            bounds = ()
        else:
            bounds = torch.tensor(self.rule_bound_mappings_by_class[cls_str][attr_name])
        stat = self.get_statistics_by_name(transformed_bbox, height, width, attr_name)
        # width = bbox[2] - bbox[0]
        if len(bounds) == 0:
            bounds = torch.tensor([stat, stat])
        
        
        assert len(bounds) == 2

        base1 = torch.abs((stat - bounds[0]))
        base2 = torch.abs((stat - bounds[1]))

        # max_base = torch.zeros(len(bbox))
        # max_base[base1 > base2] = base1[base1 > base2]
        # max_base[base1 <= base2] = base1[base1 <= base2]
        if base1 > base2:
            max_base = base1.detach()
        else:
            max_base = base2.detach()

        loss = (stat - bounds[0]) * (stat - bounds[1])/(max_base**2)
        loss = torch.clamp(loss, min=0.0, max=1.0)

        return lam * loss

    def get_group_attr_violations(self, bbox, cls, group_attr_name, by_attr_name, img_shape, lam = 0.0):
        # width = self.get_transformed_width(bbox, img_shape)
        height = self.get_transformed_height(bbox, img_shape) 
        width = self.get_transformed_width(bbox, img_shape)
        transformed_bbox = self.get_transformed_bbox(bbox, img_shape)
        cls_str = classes[cls]
        attr_key = group_attr_name + "#" + by_attr_name
        group_stat = self.get_statistics_by_name(transformed_bbox, height, width, group_attr_name)
        by_stat = self.get_statistics_by_name(transformed_bbox, height, width, by_attr_name)

        if cls_str not in self.rule_bound_mappings_by_class:
            bounds = ()
        else:    
            ratio_ls = self.rule_bound_mappings_by_class[cls_str][attr_key]
            
            bounds = torch.tensor(get_by_attr_bounds(group_stat, ratio_ls))
      
        
        # width = bbox[2] - bbox[0]
        if len(bounds) == 0:
            bounds = torch.tensor([by_stat, by_stat])
        
        
        assert len(bounds) == 2

        base1 = torch.abs((by_stat - bounds[0]))
        base2 = torch.abs((by_stat - bounds[1]))

        if base1 > base2:
            max_base = base1.detach()
        else:
            max_base = base2.detach()

        loss = (by_stat - bounds[0]) * (by_stat - bounds[1]) / (max_base ** 2)
        loss = torch.clamp(loss, min=0.0, max=1.0)

        return lam * loss

    def get_width_violations(self, bbox, cls, img_shape, lam = 0.0):
        width = self.get_transformed_width(bbox, img_shape)
        # width = bbox[2] - bbox[0]
        bounds = torch.tensor(get_stat_bounds(widths, cls))
        if len(bounds) == 0:
            bounds = torch.tensor((width, width))

        assert len(bounds) == 2

        loss = (width - bounds[0]) * (width - bounds[1])
        loss = torch.clamp(loss, min=0.0, max=1.0)

        return lam * loss

    def get_width_violations2(self, bbox, cls, img_shape, lam = 0.0, epsilon = 5):
        width = self.get_transformed_width(bbox, img_shape)
        bounds = torch.tensor(get_stat_bounds(widths, cls))
        if len(bounds) == 0:
            bounds = torch.tensor((width, width))

        assert len(bounds) == 2
        if bbox[0] > epsilon and bbox[2] < img_shape[1] - epsilon:
            loss = (width - bounds[0]) * (width - bounds[1])
            loss = torch.clamp(loss, min=0.0, max=1.0)
        else:
            loss = (width - bounds[1])# * (aspect_ratio - bounds[1])
            loss = torch.clamp(loss, min=0.0, max=1.0)

        return lam * loss

    def get_area_violations(self, bbox, cls, img_shape, lam = 0.0):
        # width = bbox[2] - bbox[0]
        # height = bbox[3] - bbox[1]
        height = self.get_transformed_height(bbox, img_shape) 
        width = self.get_transformed_width(bbox, img_shape)
        transformed_bbox = self.get_transformed_bbox(bbox, img_shape)
        area = width * height

        bounds = torch.tensor(get_area_bounds(cls, area))
        if len(bounds) == 0:
            bounds = torch.tensor((area, area))

        assert len(bounds) == 2
        # center_x, center_y of the object
        center = (transformed_bbox[0] + transformed_bbox[2]/2, transformed_bbox[1] + transformed_bbox[3]/2)
        # 2048/2, 1024/2
        # l2 norm distance the following area should be distance
        distance = torch.sqrt((center[0] - 2048/2)**2 + (center[1] - 1024/2)**2)
        loss = (distance - bounds[0]) * (distance - bounds[1])
        loss = torch.clamp(loss, min=0.0, max=1.0)

        return lam * loss

    def get_area_violations2(self, bbox, cls, img_shape, lam = 0.0, epsilon=5):
        # width = bbox[2] - bbox[0]
        # height = bbox[3] - bbox[1]
        height = self.get_transformed_height(bbox, img_shape) 
        width = self.get_transformed_width(bbox, img_shape)
        area = width * height

        bounds = torch.tensor(get_area_bounds(cls, area))
        if len(bounds) == 0:
            bounds = torch.tensor((area, area))

        assert len(bounds) == 2

        if bbox[0] > epsilon and bbox[2] < img_shape[1] - epsilon:
            loss = (area - bounds[0]) * (area - bounds[1])
            loss = torch.clamp(loss, min=0.0, max=1.0)

        else:
            loss = (bounds[0] - area)# * (aspect_ratio - bounds[1])
            loss = torch.clamp(loss, min=0.0, max=1.0)


        return lam * loss


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None, test_adaptations=False, tta_method="rule"):

        if not test_adaptations:

            x = self.extract_feat(img)
            losses = dict()

            semantic_logits = self.semantic_head(x[:4])
            loss_seg = self.semantic_head.loss(semantic_logits, gt_semantic_seg)
            losses.update(loss_seg)

            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas,
                                            self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                    *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

            sampling_results =  self.assign_result(x, proposal_list, img,
                                gt_bboxes, gt_labels, gt_bboxes_ignore)
        
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats = self.bbox_roi_extractor(
                    x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    self.train_cfg.rcnn)

            # print(f"Classes: {cls_score}\nPredicted bboxes: {bbox_pred}")
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            # print(f"BBox loss: {loss_bbox}")
            losses.update(loss_bbox)


            pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
            mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
            if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)

            if mask_feats.shape[0] > 0:
                mask_pred = self.mask_head(mask_feats)
                mask_targets = self.mask_head.get_target(
                    sampling_results, gt_masks, self.train_cfg.rcnn)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                                pos_labels)
                losses.update(loss_mask)

            # proposal_list = self.simple_test_rpn(x, img_metas,
            #                              self.test_cfg.rpn)

            # det_bboxes, det_labels = self.simple_test_bboxes(x, 
            #     img_metas, proposal_list, self.test_cfg.rcnn, rescale=False)
            # bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            # print(f"Raw result: {det_bboxes}")
            # print(f"Result: {bbox_results}\nLabels: {det_labels}")
            
            # print(f"Losses: {losses}")

            # img_shape = img_metas[0]['img_shape']
            # scale_factor = img_metas[0]['scale_factor']
            # det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            #     rois,
            #     cls_score,
            #     bbox_pred,
            #     img_shape,
            #     scale_factor,
            #     rescale=False,
            #     cfg= self.test_cfg.rcnn)

            det_bboxes, det_labels = self.simple_test_bboxes(x, img_metas, proposal_list, self.test_cfg.rcnn, rescale=False)
            print(len(det_bboxes))
            
            aspect_ratios = [ self.get_aspect_ratio_violation(box, det_labels[i], LAMBDA) for i, box in enumerate(det_bboxes) ]
            losses.update({
                "loss_aspect_ratios": aspect_ratios
            })
            
            widths = [ self.get_width_violations(box, det_labels[i], LAMBDA) for i, box in enumerate(det_bboxes) ]
            losses.update({
                "loss_widths": widths
            })
            
            heights = [ self.get_height_violations(box, det_labels[i], LAMBDA) for i, box in enumerate(det_bboxes) ]
            losses.update({
                "loss_heights": heights
            })
            
            areas = [ self.get_area_violations(box, det_labels[i], LAMBDA) for i, box in enumerate(det_bboxes) ]
            losses.update({
                "loss_areas": areas
            })

            return losses

        else:
            return self.simple_test_adaptations(img, img_metas, tta_method=tta_method, rescale=True)




    def simple_test(self, img, img_metas, proposals=None, rescale=False, eval=None):
        x = self.extract_feat(img)
        semantic_logits = self.semantic_head(x[:4])
        result = []
        if semantic_logits.shape[0] == 1:
            proposal_list = self.simple_test_rpn(x, img_metas,
                                     self.test_cfg.rpn)

            det_bboxes, det_labels = self.simple_test_bboxes(x, 
                img_metas, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        
            if eval is not None:
                       
                panoptic_mask, cat_ = self.simple_test_mask_(
                    x, img_metas, det_bboxes, det_labels, semantic_logits, rescale=rescale)
                result.append([panoptic_mask, cat_, img_metas])
        
            else:          
                bbox_results = bbox2result(det_bboxes, det_labels,
                                           self.bbox_head.num_classes)
                mask_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, semantic_logits, rescale=rescale)

                return bbox_results, mask_results
        else:
            for i in range(len(img_metas)):
                new_x = []
                for x_i in x:
                    new_x.append(x_i[i:i+1])
                proposal_list = self.simple_test_rpn(new_x, [img_metas[i]],
                                     self.test_cfg.rpn)

                assert eval is not None

                det_bboxes, det_labels = self.simple_test_bboxes(new_x, 
                    [img_metas[i]], proposal_list, self.test_cfg.rcnn, rescale=rescale)

                panoptic_mask, cat_ = self.simple_test_mask_(
                    new_x, [img_metas[i]], det_bboxes, det_labels, semantic_logits[i:i+1], rescale=rescale)

                result.append([panoptic_mask, cat_, [img_metas[i]]])

        return result

    def compute_rule_losses(self, det_bboxes, det_labels, losses, img_shape):
        # aspect_ratios = [ self.get_aspect_ratio_violation2(box, det_labels[i], img_shape, LAMBDA) for i, box in enumerate(det_bboxes) ]
        aspect_ratios = [ self.get_aspect_ratio_violation(box, det_labels[i], img_shape, LAMBDA) for i, box in enumerate(det_bboxes) ]
        losses.update({
                "loss_aspect_ratios": aspect_ratios
            })
            
        widths = [ self.get_width_violations(box, det_labels[i], img_shape, LAMBDA) for i, box in enumerate(det_bboxes) ]
        # widths = [ self.get_width_violations(box, det_labels[i], img_shape, LAMBDA) for i, box in enumerate(det_bboxes) ]
        losses.update({
            "loss_widths": widths
        })
        
        heights = [ self.get_height_violations(box, det_labels[i], img_shape, LAMBDA) for i, box in enumerate(det_bboxes) ]
        losses.update({
            "loss_heights": heights
        })
        
        areas = [ self.get_area_violations(box, det_labels[i], img_shape, LAMBDA) for i, box in enumerate(det_bboxes) ]
        # areas = [ self.get_area_violations(box, det_labels[i], LAMBDA) for i, box in enumerate(det_bboxes) ]
        losses.update({
            "loss_areas": areas
        })

    def compute_rule_losses2(self, det_bboxes, det_labels, losses, img_shape):
        # aspect_ratios = [ self.get_aspect_ratio_violation2(box, det_labels[i], img_shape, LAMBDA) for i, box in enumerate(det_bboxes) ]

        for attr in attribute_name_ls:
            # self.get_single_attr_violations()
            attr_loss = [ self.get_single_attr_violations(box, det_labels[i], attr, img_shape, LAMBDA) for i, box in enumerate(det_bboxes) ]
            losses.update({
                "loss_" + attr: attr_loss
            })
        
        for attr1 in attribute_name_ls:
            for attr2 in attribute_name_ls:
                if not attr1 == attr2:
                    group_by_loss = [ self.get_group_attr_violations(box, det_labels[i], attr1, attr2, img_shape, LAMBDA) for i, box in enumerate(det_bboxes) ]
                    losses.update({
                        "loss_" + attr1 + "_by_" + attr2: group_by_loss
                    })            
        # aspect_ratios = [ self.get_aspect_ratio_violation(box, det_labels[i], img_shape, LAMBDA) for i, box in enumerate(det_bboxes) ]
        # losses.update({
        #         "loss_aspect_ratios": aspect_ratios
        #     })
            
        # widths = [ self.get_width_violations(box, det_labels[i], img_shape, LAMBDA) for i, box in enumerate(det_bboxes) ]
        # # widths = [ self.get_width_violations(box, det_labels[i], img_shape, LAMBDA) for i, box in enumerate(det_bboxes) ]
        # losses.update({
        #     "loss_widths": widths
        # })
        
        # heights = [ self.get_height_violations(box, det_labels[i], img_shape, LAMBDA) for i, box in enumerate(det_bboxes) ]
        # losses.update({
        #     "loss_heights": heights
        # })
        
        # areas = [ self.get_area_violations(box, det_labels[i], img_shape, LAMBDA) for i, box in enumerate(det_bboxes) ]
        # # areas = [ self.get_area_violations(box, det_labels[i], LAMBDA) for i, box in enumerate(det_bboxes) ]
        # losses.update({
        #     "loss_areas": areas
        # })

    def simple_test_adaptations(self, img, img_metas, tta_method ='rule', rescale=True):
        x = self.extract_feat(img)
        semantic_logits = self.semantic_head(x[:4])
        losses = dict()
        if semantic_logits.shape[0] == 1:
            proposal_list = self.simple_test_rpn(x, img_metas,
                                     self.test_cfg.rpn)

            det_bboxes, det_labels = self.simple_test_bboxes(x, 
                img_metas, proposal_list, self.test_cfg.rcnn, rescale=rescale)
            confidence_scores = det_bboxes[:,-1]
            if tta_method == 'rule':
                if self.rule_path is None:
                    self.compute_rule_losses(det_bboxes, det_labels, losses, img_metas[0]['ori_shape'])
                else:
                    self.compute_rule_losses2(det_bboxes, det_labels, losses, img_metas[0]['ori_shape'])

            elif tta_method == 'tent':
                self.tta_loss_bbox(det_bboxes[:,0:4], losses)
            elif tta_method == "cpl":
                
                # outputs = torch.sigmoid(net(inputs_num, inputs_cat))
                if len(det_bboxes) > 0:
                    loss = conjugate_pl(det_bboxes[:,0:4], num_classes=4, input_probs=True)
                    losses['cpl_bbox_loss'] = loss/1000
                else:
                    synth_data = torch.rand([1,10], requires_grad=True)
                    losses['cpl_bbox_loss'] = torch.mean(synth_data)
            elif tta_method == "rpl":
                # outputs = torch.sigmoid(net(inputs_num, inputs_cat))
                if len(det_bboxes) > 0:
                    loss = robust_pl(det_bboxes[:,0:4], input_probs=True)
                    losses['rpl_bbox_loss'] = loss/1000 + 1
                else:
                    synth_data = torch.rand([1,10], requires_grad=True)
                    losses['rpl_bbox_loss'] = torch.mean(synth_data)
            elif tta_method == 'memo':
                return det_bboxes[:,0:4]
            elif tta_method == "norm":
                synth_data = torch.rand([1,10], requires_grad=True)
                losses['rpl_bbox_loss'] = torch.mean(synth_data)
    
            # if eval is not None:
                       
            #     panoptic_mask, cat_ = self.simple_test_mask_(
            #         x, img_metas, det_bboxes, det_labels, semantic_logits, rescale=rescale)
            #     result.append([panoptic_mask, cat_, img_metas])
        
            # else:          
            #     bbox_results = bbox2result(det_bboxes, det_labels,
            #                                self.bbox_head.num_classes)
            #     mask_results = self.simple_test_mask(
            #         x, img_metas, det_bboxes, det_labels, semantic_logits, rescale=rescale)

            #     return bbox_results, mask_results
        else:
            for i in range(len(img_metas)):
                new_x = []
                for x_i in x:
                    new_x.append(x_i[i:i+1])
                proposal_list = self.simple_test_rpn(new_x, [img_metas[i]],
                                     self.test_cfg.rpn)

                # assert eval is not None

                det_bboxes, det_labels = self.simple_test_bboxes(new_x, 
                    [img_metas[i]], proposal_list, self.test_cfg.rcnn, rescale=rescale)

                if tta_method == 'rule':
                    if self.rule_path is None:
                        self.compute_rule_losses(det_bboxes, det_labels, losses, img_metas[0]['ori_shape'])
                    else:
                        self.compute_rule_losses2(det_bboxes, det_labels, losses, img_metas[0]['ori_shape'])
                elif tta_method == 'tent':
                    self.tta_loss_bbox(det_bboxes, losses)


                # panoptic_mask, cat_ = self.simple_test_mask_(
                #     new_x, [img_metas[i]], det_bboxes, det_labels, semantic_logits[i:i+1], rescale=rescale)

                # result.append([panoptic_mask, cat_, [img_metas[i]]])

        return losses

    def aug_test(self,):
        pass


    def simple_test_rpn(self, x, img_metas, rpn_test_cfg):
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_metas, rpn_test_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def simple_test_bboxes(self,
                    x,
                    img_metas,
                    proposals,
                    rcnn_test_cfg,
                    rescale=False):

        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def simple_test_mask(self,
                  x,
                  img_metas,
                  det_bboxes,
                  det_labels,
                  semantic_logits, 
                  rescale=False):

        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)

            segm_result = self.mask_head.get_seg_masks(mask_pred, _bboxes,
                                                       det_labels,
                                                       self.test_cfg.rcnn,
                                                       ori_shape, scale_factor,
                                                       rescale)
        return segm_result

    def simple_test_mask_(self,
                  x,
                  img_metas,
                  det_bboxes,
                  det_labels,
                  semantic_logits, 
                  rescale=False):

        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if rescale:
            ref_size = (np.int(np.round(ori_shape[0]*scale_factor)), 
                        np.int(np.round(ori_shape[1]*scale_factor)))
        else:
            ref_size = (np.int(np.round(ori_shape[0])), 
                        np.int(np.round(ori_shape[1])))
        semantic_logits = F.interpolate(semantic_logits, size=ref_size, 
                                   mode="bilinear", align_corners=False)   
        sem_pred = torch.argmax(semantic_logits, dim=1)[0]
        panoptic_mask = torch.zeros_like(sem_pred, dtype=torch.long)
        cat = [255]
        if det_bboxes.shape[0] == 0:
            intermediate_logits = semantic_logits[0, :self.num_stuff] 
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            confidence = det_bboxes[:,4]
            idx = torch.argsort(confidence, descending=True)
            bbx_inv = invert_roi_bbx(_bboxes[:, :4], 
                      tuple(mask_pred.shape[2:]), ref_size)
            bbx_idx = torch.arange(0, det_bboxes.size(0), 
                      dtype=torch.long, device=det_bboxes.device)
            
            mask_pred = roi_sampling(mask_pred, bbx_inv, bbx_idx, 
                        ref_size, padding="zero")
            ML_A = mask_pred.new_zeros(mask_pred.shape[0], mask_pred.shape[-2], 
                                             mask_pred.shape[-1])
            ML_B = ML_A.clone()             
            occupied = torch.zeros_like(sem_pred, dtype=torch.bool)
            i = 0 
            for id_i in idx:
                label_i = det_labels[id_i] 
                mask_pred_i = mask_pred[id_i, label_i+1, :, :]
                mask_i = (mask_pred_i.sigmoid() > self.test_cfg.rcnn.mask_thr_binary) 
                mask_i = mask_i.type(torch.bool)
                intersection = occupied & mask_i
                if intersection.float().sum() / mask_i.float().sum() > self.test_cfg.panoptic.overlap_thr:
                    continue

                mask_i = mask_i ^ intersection
                occupied += mask_i

                y0 = max(int(_bboxes[id_i, 1] + 1), 0)
                y1 = min(int((_bboxes[id_i, 3] - 1).round() + 1), ref_size[0])
                x0 = max(int(_bboxes[id_i, 0] + 1), 0)
                x1 = min(int((_bboxes[id_i, 2] - 1).round() + 1), ref_size[1])

                ML_A[i] = 4 * mask_pred_i
                ML_B[i, y0: y1, x0: x1] = semantic_logits[0, label_i + self.num_stuff, y0: y1, x0: x1]
                cat.append(label_i.item() + self.num_stuff)
                i = i + 1 

            ML_A = ML_A[:i]
            ML_B = ML_B[:i]
            FL = (ML_A.sigmoid() + ML_B.sigmoid())*(ML_A + ML_B)
            intermediate_logits = torch.cat([semantic_logits[0, :self.num_stuff], FL], dim=0)

        cat = torch.tensor(cat, dtype=torch.long)
        intermediate_mask = torch.argmax(F.softmax(intermediate_logits, dim=0), dim=0) + 1
        intermediate_mask = intermediate_mask - self.num_stuff
        intermediate_mask[intermediate_mask <= 0] = 0         
        unique = torch.unique(intermediate_mask) 
        ignore_val = intermediate_mask.max().item() + 1
        ignore_arr = torch.ones((ignore_val,), dtype=unique.dtype, device=unique.device) * ignore_val
        total_unique = unique.shape[0]
        ignore_arr[unique] = torch.arange(total_unique).cuda(ignore_arr.device)  
        panoptic_mask = ignore_arr[intermediate_mask]
        panoptic_mask[intermediate_mask == ignore_val] = 0 

        cat_ = cat[unique].long()
        sem_pred[panoptic_mask > 0] = self.num_stuff
        sem_pred[sem_pred >= self.num_stuff] = self.num_stuff
        cls_stuff, area = torch.unique(sem_pred, return_counts=True)
        cls_stuff[area < self.test_cfg.panoptic.min_stuff_area] = self.num_stuff
        cls_stuff = cls_stuff[cls_stuff!=self.num_stuff]     

        tmp = torch.ones((self.num_stuff + 1,), dtype=cls_stuff.dtype, device=cls_stuff.device) * self.num_stuff
        tmp[cls_stuff] = torch.arange(cls_stuff.shape[0]).cuda(tmp.device)  
        new_sem_pred = tmp[sem_pred]
        cat_ = torch.cat((cat_, cls_stuff.cpu().long()), -1)   
        bool_mask = new_sem_pred != self.num_stuff   
        panoptic_mask[bool_mask] = new_sem_pred[bool_mask] + total_unique 

        return panoptic_mask.cpu(), cat_.cpu()
