import os

# os.chdir('..')
import tensorflow as tf

from samples.coco import coco
import eb_coco
from preprocess import preprocess
from preprocess import augmentation as aug
# from training import train_model
from training_jw import train_model
from model import mask_rcnn_functional
from common.utils import tf_limit_gpu_memory

import copy
import pprint
import numpy as np
import argparse

def show_mrtf2_coco(dataset):
    print("dataset info: ", type(dataset), len(dataset))
    datapoint_num_elts = len(dataset[0])

    np.save('eb_dataset_0_0', dataset[0][0])
    np.save('eb_dataset_0_1', dataset[0][1])

    for ix in range(datapoint_num_elts):
        pprint.pprint(dataset[0][ix].shape)
        if ix == 2:
            print(dataset[0][ix])


def main():
    parser = argparse.ArgumentParser("mask rcnn tf2 training")    
    parser.add_argument("--backbone_name", type=str)
    parser.add_argument("--weights_file_path", type=str, default=None)
    args = parser.parse_args()
    
    # Limit GPU memory for tensorflow container
    # tf_limit_gpu_memory(tf, 10000)
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"

    # tf_limit_gpu_memory(tf, 20000)

    from common.config import CONFIG

    CONFIG['backbone'] = args.backbone_name
    CONFIG.update(eb_coco.EB_COCO_CONFIG)

    ############# learning rate
    CONFIG['callback']['histogram_freq'] = 500
    CONFIG['callback']['profile_batch'] = (500, 501)


    # Init Mask-RCNN model
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    strategy = None
    num_gpus = CONFIG['gpu_num']
    if num_gpus>1:
        devices = tf.config.experimental.list_physical_devices('GPU')
        devices_names = [d.name.split('e:')[1] for d in devices]
        assert len(devices_names)>=num_gpus, f"Host machine has {len(devices_names)} GPUs. Provide gpu_num <= available GPUs."
        devices_names = devices_names[:num_gpus]
        strategy = tf.distribute.MirroredStrategy(devices=devices_names)
        CONFIG['optimizer_kwargs']['learning_rate'] = num_gpus*CONFIG['optimizer_kwargs']['learning_rate']
        # CONFIG['batch_size'] = 4

    if strategy:
        with strategy.scope():
            model = mask_rcnn_functional(config=CONFIG)
    else:
        model = mask_rcnn_functional(config=CONFIG)

    base_dataset_dir = CONFIG["base_dataset_dir"]

    train_dataset = coco.CocoDataset(dataset_dir=base_dataset_dir,
                                     subset='train',
                                     year=2017,
                                     auto_download=False,
                                     preprocess_transform=None,
                                    #  augmentation=aug.get_training_augmentation(),
                                     class_ids=[1, 2, 3],
                                     **CONFIG
                                     )

    show_mrtf2_coco(train_dataset)

    val_dataset = coco.CocoDataset(dataset_dir=base_dataset_dir,
                                subset='val',
                                year=2017,
                                auto_download=False,
                                preprocess_transform=None,
                                class_ids=[1, 2, 3],
                                **CONFIG
                                )

    print("length of train, val dataset: ",len(train_dataset), len(val_dataset))

    train_model(model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=CONFIG,
                weights_path=args.weights_file_path,
                strategy=strategy)

if __name__=="__main__":
    main()
