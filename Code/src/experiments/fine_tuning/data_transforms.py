import math
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms
from torchvision.transforms import functional as functional


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_FILL = tuple(round(255 * value) for value in IMAGENET_MEAN)


class DeiTRandAugment:
    """Local equivalent of timm's rand-m9-mstd0.5-inc1 policy."""

    operations = (
        "AutoContrast", "Equalize", "Invert", "Rotate",
        "Posterize", "Solarize", "SolarizeAdd", "Color",
        "Contrast", "Brightness", "Sharpness", "ShearX",
        "ShearY", "TranslateX", "TranslateY",
    )

    def __call__(self, image):
        # RandAugment selects two operations with replacement.
        operations = np.random.choice(self.operations, 2, replace=True)
        for operation in operations:
            if random.random() > 0.5:
                continue

            magnitude = min(10.0, max(0.0, random.gauss(9.0, 0.5)))
            image = self._apply(image, operation, magnitude)

        return image

    @staticmethod
    def _signed(value):
        return -value if random.random() > 0.5 else value

    def _apply(self, image, operation, magnitude):
        if operation == "AutoContrast":
            return ImageOps.autocontrast(image)
        if operation == "Equalize":
            return ImageOps.equalize(image)
        if operation == "Invert":
            return ImageOps.invert(image)
        if operation == "Rotate":
            degrees = self._signed(magnitude / 10.0 * 30.0)
            return image.rotate(degrees, resample=Image.Resampling.BICUBIC, fillcolor=IMAGENET_FILL)
        if operation == "Posterize":
            bits = 4 - int(magnitude / 10.0 * 4)
            return ImageOps.posterize(image, bits)
        if operation == "Solarize":
            threshold = 256 - min(256, int(magnitude / 10.0 * 256))
            return ImageOps.solarize(image, threshold)
        if operation == "SolarizeAdd":
            addition = min(128, int(magnitude / 10.0 * 110))
            lookup = [min(255, value + addition) if value < 128 else value for value in range(256)]
            return image.point(lookup * 3)

        if operation in ("Color", "Contrast", "Brightness", "Sharpness"):
            change = magnitude / 10.0 * 0.9
            factor = max(0.1, 1.0 + self._signed(change))
            enhancer = {"Color": ImageEnhance.Color, "Contrast": ImageEnhance.Contrast, 
                        "Brightness": ImageEnhance.Brightness, "Sharpness": ImageEnhance.Sharpness}[operation]
            return enhancer(image).enhance(factor)

        if operation in ("ShearX", "ShearY"):
            shear = self._signed(magnitude / 10.0 * 0.3)
            matrix = ((1, shear, 0, 0, 1, 0) if operation == "ShearX" else (1, 0, 0, shear, 1, 0))
        else:
            translation = self._signed(magnitude / 10.0 * 0.45)
            matrix = ((1, 0, translation * image.width, 0, 1, 0)
                      if operation == "TranslateX"
                      else (1, 0, 0, 0, 1, translation * image.height))

        return image.transform(image.size, Image.Transform.AFFINE, matrix, 
                               resample=Image.Resampling.BICUBIC, fillcolor=IMAGENET_FILL)


class DeiTRandomResizedCrop:
    """Random resized crop using the sampling procedure from timm."""

    def __init__(self, image_size):
        self.image_size = image_size
        self.scale = (0.08, 1.0)
        self.ratio = (3 / 4, 4 / 3)

    def __call__(self, image):
        area = image.width * image.height
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            width = int(round(math.sqrt(target_area * aspect_ratio)))
            height = int(round(math.sqrt(target_area / aspect_ratio)))

            if width <= image.width and height <= image.height:
                top = random.randint(0, image.height - height)
                left = random.randint(0, image.width - width)
                return functional.resized_crop(image, top, left, height, width, 
                                               (self.image_size, self.image_size), transforms.InterpolationMode.BICUBIC)

        # Use the same centered fallback as timm when no random crop fits.
        input_ratio = image.width / image.height
        if input_ratio < self.ratio[0]:
            width = image.width
            height = int(round(width / self.ratio[0]))
        elif input_ratio > self.ratio[1]:
            height = image.height
            width = int(round(height * self.ratio[1]))
        else:
            width = image.width
            height = image.height

        top = (image.height - height) // 2
        left = (image.width - width) // 2
        return functional.resized_crop(image, top, left, height, width, 
                                       (self.image_size, self.image_size), transforms.InterpolationMode.BICUBIC)


class DeiTRandomErasing:
    """Per-pixel random erasing with timm's exact sampling bounds."""

    def __call__(self, images):
        if images.ndim == 3:
            return self._erase(images)

        for image in images:
            self._erase(image)
        return images

    def _erase(self, image):
        if random.random() > 0.25:
            return image

        channels, image_height, image_width = image.shape
        area = image_height * image_width
        log_ratio = (math.log(0.3), math.log(1 / 0.3))

        for _ in range(10):
            target_area = random.uniform(0.02, 1 / 3) * area
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            height = int(round(math.sqrt(target_area * aspect_ratio)))
            width = int(round(math.sqrt(target_area / aspect_ratio)))

            if width < image_width and height < image_height:
                top = random.randint(0, image_height - height)
                left = random.randint(0, image_width - width)
                image[:, top:top + height, left:left + width] = torch.empty((channels, height, width), dtype=image.dtype, 
                                                                            device=image.device).normal_()
                break

        return image


class BatchMixupCutmix:
    """timm-style batch Mixup/CutMix with label smoothing."""

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, images, targets):
        use_cutmix = np.random.rand() < 0.5
        alpha = 1.0 if use_cutmix else 0.8
        mixing = float(np.random.beta(alpha, alpha))

        if use_cutmix:
            ratio = math.sqrt(1 - mixing)
            height, width = images.shape[-2:]
            cut_height = int(height * ratio)
            cut_width = int(width * ratio)
            center_y = np.random.randint(0, height)
            center_x = np.random.randint(0, width)
            top = np.clip(center_y - cut_height // 2, 0, height)
            bottom = np.clip(center_y + cut_height // 2, 0, height)
            left = np.clip(center_x - cut_width // 2, 0, width)
            right = np.clip(center_x + cut_width // 2, 0, width)

            images[:, :, top:bottom, left:right] = images.flip(0)[:, :, top:bottom, left:right]
            mixing = 1 - ((bottom - top) * (right - left) / (height * width))
        else:
            flipped_images = images.flip(0).mul_(1 - mixing)
            images.mul_(mixing).add_(flipped_images)

        off_value = 0.1 / self.num_classes
        on_value = 1 - 0.1 + off_value
        targets = torch.full((targets.shape[0], self.num_classes), off_value, 
                             device=targets.device).scatter_(1, targets.view(-1, 1), on_value)
        flipped_targets = targets.flip(0)
        targets = targets * mixing + flipped_targets * (1 - mixing)
        return images, targets


def build_training_transform(image_size):
    return transforms.Compose([DeiTRandomResizedCrop(image_size), transforms.RandomHorizontalFlip(), 
                               DeiTRandAugment(), transforms.PILToTensor()])


def build_evaluation_transform(image_size):
    resize_size = int(256 / 224 * image_size)
    return transforms.Compose([transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC), 
                               transforms.CenterCrop(image_size), transforms.ToTensor(), 
                               transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
