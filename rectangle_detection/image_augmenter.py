import imgaug.augmenters as iaa

seq = iaa.Sequential([
    iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
    iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 255)))
])
