import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# ImageNet code should change this value
IMAGE_SIZE = 32
import torch

def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
#   return np.random.uniform(low=0.1, high=n)
  return n


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  # return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
  #                          Image.AFFINE, (1, level, 0, 0, 1, 0),
  #                          resample=Image.BILINEAR)
  return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)

def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  # return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
  #                          Image.AFFINE, (1, 0, 0, level, 1, 0),
  #                          resample=Image.BILINEAR)
  return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), pil_img.size[0] / 3)
  if np.random.random() > 0.5:
    level = -level
  # return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
  #                          Image.AFFINE, (1, 0, level, 0, 1, 0),
  #                          resample=Image.BILINEAR)
  return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  # level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  level = int_parameter(sample_level(level), pil_img.size[0] / 3)
  if np.random.random() > 0.5:
    level = -level
  # return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
  #                          Image.AFFINE, (1, 0, 0, 0, 1, level),
  #                          resample=Image.BILINEAR)
  return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]


def create_copy_for_img_dict(input_img):
    image = dict()
    for key in input_img:
        if type(input_img[key]) is torch.Tensor or type(input_img[key]) is np.ndarray:
            image[key] = input_img[key].copy()
        else:
            image[key] = input_img[key]

    return image

def image_aug(input_img, preprocess):
    """Perform AugMix augmentations and compute mixture.
    Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.
    Returns:
    mixed: Augmented and mixed image.
    """
    if type(input_img) is dict:
        image = create_copy_for_img_dict(input_img)
    else:
        image = input_img

    mixture_width = 3
    mixture_depth = -1
    aug_severity = 3
    # preprocess = transforms.Compose(
    #             [transforms.ToTensor()])
    aug_list = augmentations_all
    #   if args.all_ops:
    # aug_list = augmentations.augmentations_all

    ws = np.float32(np.random.dirichlet([1] * mixture_width))
    #   m = np.float32(np.random.beta(1, 1))
    m = 0.4

    
    for i in range(mixture_width):
        if type(input_img) is dict:
            image_aug = create_copy_for_img_dict(input_img)
        else:
            image_aug = image.copy()
        depth = mixture_depth if mixture_depth > 0 else np.random.randint(
            1, 4)
    for _ in range(depth):
        op = np.random.choice(aug_list)
        if type(input_img) is dict:
            image_aug_img = op(Image.fromarray(image_aug['img']), 3)
            image_aug = create_copy_for_img_dict(input_img)
            image_aug['img'] = np.array(image_aug_img)
        else:
            image_aug = op(image_aug, 3)
    # Preprocessing commutes since all coefficients are convex
    if type(input_img) is dict:
        processed_img = preprocess(image)
        aug_processed_img = preprocess(image_aug)
        mix = torch.zeros_like(processed_img['img'].data)
        mix += ws[i] * aug_processed_img['img'].data
        mixed_img = (1 - m) * processed_img['img'].data + m * mix
        mixed = create_copy_for_img_dict(processed_img)
        mixed['img']._data = mixed_img
        # mixed = mixed_img['img']
    else:
        mix = torch.zeros_like(preprocess(image))
        mix += ws[i] * preprocess(image_aug)
        mixed = (1 - m) * preprocess(image) + m * mix
    return mixed