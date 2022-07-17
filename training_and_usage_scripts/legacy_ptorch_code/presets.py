#from torchvision.transforms import autoaugment, transforms
from torchvision.transforms import transforms


class ClassificationPresetTrain:
    def __init__(self, crop_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), hflip_prob=0.5,
                 auto_augment_policy=None, random_erase_prob=0.0):
        trans = [transforms.RandomResizedCrop(crop_size)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
            trans.append(autoaugment.AutoAugment(policy=aa_policy))
        trans.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(self, crop_size, resize_size=256, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), random_erase_prob=0.0, random_erase_scale = (0.02,0.33), random_erase_ratio = (0.3,3.3), random_erase_val = 0):

        trans = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob,
                                                  scale=random_erase_scale,
                                                  ratio=random_erase_ratio,
                                                  value=random_erase_val
                                                 ))
        self.transforms = transforms.Compose(trans)
        

    def __call__(self, img):
        return self.transforms(img)
#{"mode":"full","isActive":false}