import alphapose
import argparse
from os.path import dirname, join

ap_path = dirname(alphapose.__file__)
out_path = join(dirname(__file__), "alphapose_out")
print(ap_path)

configs = {
    'default' : argparse.Namespace(
        model_cfg= join(ap_path, '../configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml'),
        checkpoint= join(ap_path, '../pretrained_models/halpe26_fast_res50_256x192.pth'),
        debug=False,
        detbatch=5,
        detector='yolo',
        detfile='',
        eval=False,
        flip=False,
        format=None,
        gpus='0',
        #inputimg='/home/guest/AlphaPose/examples/my_demo/human.jpg', #human_4m.jpg'
        inputlist='',
        inputpath='',
        min_box_area=0,
        outputpath= out_path,
        pose_flow=False,
        pose_track=False,
        posebatch=64,
        profile=True, 
        qsize=1024,
        save_img=True,
        save_video=False,
        showbox=False,
        sp=False,
        video='',
        vis=False,
        vis_fast=False,
        webcam=-1,
    ),
}