import random

from PIL import ImageFilter, ImageOps
from torchvision import transforms


class GaussianBlur:
    """Apply Gaussian blur with a randomly sampled radius."""

    def __init__(self, probability=0.5, radius_min=0.1, radius_max=2.0):
        self.probability = probability
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, image):
        if random.random() > self.probability:
            return image

        radius = random.uniform(
            self.radius_min,
            self.radius_max,
        )

        return image.filter(
            ImageFilter.GaussianBlur(radius=radius)
        )


class Solarization:
    """Apply solarization with the specified probability."""

    def __init__(self, probability=0.2):
        self.probability = probability

    def __call__(self, image):
        if random.random() < self.probability:
            return ImageOps.solarize(image)

        return image


class DINOMultiCropTransform:
    """
    Create two global views and multiple local views of one image.

    The labels are not used during DINO pretraining.
    """

    def __init__(
        self,
        global_crop_size=224,
        local_crop_size=96,
        global_crop_scale=(0.4, 1.0),
        local_crop_scale=(0.05, 0.4),
        num_local_crops=6,
    ):
        self.num_local_crops = num_local_crops

        color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1,
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        # First global crop.
        self.global_transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crop_size,
                scale=global_crop_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            color_jitter,
            GaussianBlur(probability=1.0),
            normalize,
        ])

        # Second global crop.
        self.global_transform_2 = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crop_size,
                scale=global_crop_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            color_jitter,
            GaussianBlur(probability=0.1),
            Solarization(probability=0.2),
            normalize,
        ])

        # Local crops.
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crop_size,
                scale=local_crop_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            color_jitter,
            GaussianBlur(probability=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = [
            self.global_transform_1(image),
            self.global_transform_2(image),
        ]

        for _ in range(self.num_local_crops):
            crops.append(self.local_transform(image))

        return crops
