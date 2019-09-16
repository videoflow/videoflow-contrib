# Licensed under the Apache License, Version 2.0
# Contains code taken from: https://github.com/dmlc/gluon-cv/blob/137ebb41962ac75975cf8923b58d884c880198ce/scripts/re-id/baseline/test.py

from mxnet.gluon.data.vision import transforms


def get_transform():
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(size=(128, 384), interpolation=1),
        transforms.ToTensor(),
        normalizer,
    ])
    return transform
