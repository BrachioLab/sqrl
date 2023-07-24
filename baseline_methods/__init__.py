available_tta_methods=["rule", "memo", "tent", "cpl", "rpl", "norm"]
from .baseline import memo_loss as memo_loss
from .baseline import l2_consistency_loss as l2_consistency_loss
from .baseline import entropy_classification_loss as entropy_classification_loss
from .baseline import robust_pl as robust_pl
from .baseline import conjugate_pl as conjugate_pl
from .augmentations import image_aug, create_copy_for_img_dict
