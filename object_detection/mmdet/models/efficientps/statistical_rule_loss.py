import torch

LAMBDA = 0.001
classes = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


aspect_ratios = {"bicycle": [0.185185179, 12.5], "traffic sign": [0.0313588865, 27.0], "traffic light": [0.0792602375, 9.0], "car": [0.0507614203, 20.0], "person": [0.318181813, 23.2000008], "rider": [0.555555582, 5.76000023], "motorcycle": [0.141666666, 5.21428585], "bus": [0.173515975, 8.29166698], "truck": [0.212017581, 6.41891909], "caravan": [0.456521749, 2.36543202], "trailer": [0.283950627, 3.28512406]}

widths = {"bicycle": [2.0, 677.0], "traffic sign": [2.0, 1590.0], "traffic light": [2.0, 2047.0], "car": [3.0, 1147.0], "person": [2.0, 503.0], "rider": [5.0, 400.0], "motorcycle": [3.0, 834.0], "bus": [8.0, 1423.0], "truck": [12.0, 2047.0], "caravan": [35.0, 623.0], "trailer": [31.0, 369.0]}

heights = {"bicycle": [5.0, 543.0], "traffic sign": [2.0, 626.0], "traffic light": [4.0, 412.0], "car": [3.0, 1024.0], "person": [5.0, 914.0], "rider": [5.0, 639.0], "motorcycle": [4.0, 501.0], "bus": [6.0, 861.0], "truck": [11.0, 895.0], "caravan": [21.0, 958.0], "trailer": [18.0, 795.0]}

areas = {"bicycle": [[[0, 57828.6], [4.47213595499958, 1063.6245813255728]], [[57828.6, 115641.2], [60.47520152922188, 941.2625829172218]], [[115641.2, 173453.8], [248.91765706755317, 924.9963513441553]], [[173453.8, 231266.4], [126.30122723077555, 869.6599910309776]], [[231266.4, 2097152], [113.58917201916739, 556.7463515821186]]], "traffic sign": [[[0, 124197.6], [10.404326023342406, 1124.3598178519187]], [[124197.6, 248383.2], [345.52930411182206, 898.5683056952321]], [[248383.2, 372568.80000000005], [354.46085820581095, 354.46085820581095]], [[372568.80000000005, 496754.4], [367.4479554984624, 367.4479554984624]], [[496754.4, 2097152], [393.80102843949, 393.80102843949]]], "traffic light": [[[0, 168682.4], [31.184932259025352, 1104.3922536852565]], [[168682.4, 337352.8], [414.86443568953945, 524.9695229249028]], [[337352.8, 506023.19999999995], []], [[506023.19999999995, 674693.6], []], [[674693.6, 2097152], [306.0004084964594, 306.0004084964594]]], "car": [[[0, 145398.6], [6.5, 1045.038396423787]], [[145398.6, 290782.2], [9.433981132056603, 927.4034990229442]], [[290782.2, 436165.80000000005], [109.65856099730654, 847.3678363025116]], [[436165.80000000005, 581549.4], [109.32977636490436, 774.785938179056]], [[581549.4, 2097152], [57.019733426244635, 740.0]]], "person": [[[0, 90156.8], [3.2015621187164243, 1100.763939271268]], [[90156.8, 180289.6], [8.06225774829855, 938.0767559213905]], [[180289.6, 270422.4], [35.21718330588067, 905.6026998634666]], [[270422.4, 360555.2], [276.02400257948585, 832.5998438625844]], [[360555.2, 2097152], [104.59684507670391, 351.7786946362727]]], "rider": [[[0, 45315.6], [32.26840560052511, 1019.6147556798106]], [[45315.6, 90586.2], [43.83206588788623, 953.368370568271]], [[90586.2, 135856.8], [153.43158084305853, 867.0784278253035]], [[135856.8, 181127.4], [64.66065264130884, 907.0551251164396]], [[181127.4, 2097152], [89.04493247793498, 873.105091040019]]], "motorcycle": [[[0, 44716.8], [14.7648230602334, 1013.6547735792498]], [[44716.8, 89415.6], [69.35776813018136, 975.2497116123644]], [[89415.6, 134114.40000000002], [57.56083738098326, 903.0566150579929]], [[134114.40000000002, 178813.2], [33.094561486745825, 866.3420225292087]], [[178813.2, 2097152], [598.0974000946669, 808.5001546072826]]], "bus": [[[0, 135048.0], [38.30469945058961, 1021.4029567217827]], [[135048.0, 270042.0], [139.89013546351293, 879.8377691370154]], [[270042.0, 405036.0], [567.0282179927203, 773.9266761134417]], [[405036.0, 540030.0], [289.8309852310481, 690.0371366817876]], [[540030.0, 2097152], [514.7632951172801, 637.2332775365705]]], "truck": [[[0, 177785.2], [41.6293165929973, 1052.5549866871563]], [[177785.2, 355438.4], [167.20047846821492, 900.9252188722436]], [[355438.4, 533091.6000000001], [550.4682097996214, 787.9961928842042]], [[533091.6000000001, 710744.8], [476.94994496278116, 714.915554453811]], [[710744.8, 2097152], [22.005681084665387, 417.94048619390776]]], "caravan": [[[0, 92727.6], [22.588713996153036, 980.060457318833]], [[92727.6, 184489.2], [265.7277742352124, 862.3770926920542]], [[184489.2, 276250.80000000005], [205.5024330756208, 553.9072575801837]], [[276250.80000000005, 368012.4], []], [[368012.4, 2097152], [725.7280826866216, 821.1633515933355]]], "trailer": [[[0, 39640.8], [73.824115301167, 1013.9483468106253]], [[39640.8, 78597.6], [373.71011492867035, 936.8608487923914]], [[78597.6, 117554.40000000001], [386.87465670420954, 874.6913741429031]], [[117554.40000000001, 156511.2], [420.1743685661942, 807.3755322029521]], [[156511.2, 2097152], [862.1915390445444, 902.0034645166281]]]}


aspect_ratio_bounds = [
    (0.735294104, 5.73333311),
    (0.678571403, 3.83999991),
    (0.263157904, 3.67010307),
    (0.251184821, 4.0869565),
    (0.173515975, 8.29166698),
    (),
    (0.239999995, 4.0625),
    (0.354838699, 5.5)
]

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

def get_aspect_ratio_violation(bbox, cls, lam = 0.0):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    aspect_ratio = height / (width + 0.001)

    bounds = torch.tensor(aspect_ratio_bounds[cls])
    if len(bounds) == 0:
        bounds = torch.tensor((aspect_ratio, aspect_ratio))

    assert len(bounds) == 2

    loss = (aspect_ratio - bounds[0]) * (aspect_ratio - bounds[1])
    loss = torch.clamp(loss, min=0.0, max=1.0)
    # print("Aspect Ratio Loss:", loss, aspect_ratio, bounds)
    # loss = torch.tanh(loss)
    
    return lam * loss

def get_aspect_ratio_violation2(bbox, cls, img_shape, lam = 0.0, epsilon=5):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    aspect_ratio = height / (width + 0.001)

    bounds = torch.tensor(aspect_ratio_bounds[cls])
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

def get_height_violations(bbox, cls, lam = 0.0):
    height = bbox[3] - bbox[1]
    bounds = torch.tensor(get_stat_bounds(heights, cls))
    if len(bounds) == 0:
        bounds = torch.tensor((height, height))

    assert len(bounds) == 2

    loss = (height - bounds[0]) * (height - bounds[1])
    loss = torch.clamp(loss, min=0.0, max=1.0)

    return lam * loss

def get_width_violations(bbox, cls, lam = 0.0):
    width = bbox[2] - bbox[0]
    bounds = torch.tensor(get_stat_bounds(widths, cls))
    if len(bounds) == 0:
        bounds = torch.tensor((width, width))

    assert len(bounds) == 2

    loss = (width - bounds[0]) * (width - bounds[1])
    loss = torch.clamp(loss, min=0.0, max=1.0)

    return lam * loss

def get_width_violations2(bbox, cls, img_shape, lam = 0.0, epsilon = 5):
    width = bbox[2] - bbox[0]
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

def get_area_violations(bbox, cls, lam = 0.0):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height

    bounds = torch.tensor(get_area_bounds(cls, area))
    if len(bounds) == 0:
        bounds = torch.tensor((area, area))

    assert len(bounds) == 2

    loss = (area - bounds[0]) * (area - bounds[1])
    loss = torch.clamp(loss, min=0.0, max=1.0)

    return lam * loss

def get_area_violations2(bbox, cls, img_shape, lam = 0.0, epsilon=5):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
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
