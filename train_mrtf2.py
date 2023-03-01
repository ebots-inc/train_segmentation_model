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
    parser.add_argument("--use_multi_gpus", type=bool, default=False)
    args = parser.parse_args()
    
    # Limit GPU memory for tensorflow container
    # tf_limit_gpu_memory(tf, 10000)
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"

    tf_limit_gpu_memory(tf, 20000)

    from common.config import CONFIG

    CONFIG['backbone'] = args.backbone_name
    CONFIG.update(eb_coco.EB_COCO_CONFIG)

    CONFIG['callback']['checkpoints_dir'] = os.getenv('MRTF2_SAVE_PATH', ",")
    ############# learning rate
    CONFIG['optimizer_kwargs']['learning_rate'] = 3*1e-4
    #
    CONFIG['callback']['histogram_freq'] = 500
    CONFIG['callback']['profile_batch'] = (500, 501)

    # CONFIG['batch_size'] = 2
    # CONFIG['images_per_gpu'] = 2
    # Init Mask-RCNN model
    strategy = None
    if args.use_multi_gpus:
        num_gpus = CONFIG['gpu_num']
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
        # if args.weights_file_path is not None:
        #     model.load_weights(args.weights_file_path)

    # You can also download dataset with auto_download=True argument
    # It will be downloaded and unzipped in dataset_dir
    # base_dir = r'<COCO_PATH>/coco2017'
    # base_dir = '/hdd/John/datasets/coco'
    base_dir = '/home/wjohn/work/labeling/sama/deliveries/211130_132447/coco'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    dataset_config = copy.deepcopy(CONFIG)
    # dataset_config['batch_size'] = 8
    # dataset_config['images_per_gpu'] = 8
    print("Gpu batch size", CONFIG["batch_size"], dataset_config['batch_size'])
    train_dataset = coco.CocoDataset(dataset_dir=base_dir,
                                     subset='train',
                                     year=2017,
                                     auto_download=False,
                                     preprocess_transform=None,
                                    #  augmentation=aug.get_training_augmentation(),
                                     class_ids=[1, 2, 3],
                                     **dataset_config
                                     )

    show_mrtf2_coco(train_dataset)

    val_dataset = coco.CocoDataset(dataset_dir=base_dir,
                                subset='train',
                                year=2017,
                                auto_download=False,
                                preprocess_transform=None,
                                class_ids=[1, 2, 3],
                                **dataset_config
                                )

    print("length of train, val dataset: ",len(train_dataset), len(val_dataset))

    train_model(model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=CONFIG,
                weights_path=None,
                strategy=strategy)

if __name__=="__main__":
    main()
