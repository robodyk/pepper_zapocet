from os.path import dirname, join

package_dir = dirname(__file__)
mmdet_path = '/home/martin/models/mmdetection'
mmdet_config_path = join(mmdet_path, 'configs/rtmdet/')
checkpoint_dir = package_dir

configs = {
    'default': {
        'model' : {
            'config' : join(mmdet_config_path, 'rtmdet-ins_s_8xb32-300e_coco.py'),
            'checkpoint' : join(checkpoint_dir, 'rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth'),
            'device': 'cuda:0'
        },
        'out_dir' : join(package_dir, 'rtmdet_out'),
        'score_threshold' : 0.2
    },
    'tiny': {
        'model' : {
            'config' : join(mmdet_config_path, 'rtmdet-ins_tiny_8xb32-300e_coco.py'),
            'checkpoint' : join(checkpoint_dir, 'rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth'),
            'device': 'cuda:0'
        },
        'out_dir' : join(package_dir, 'rtmdet_out'),
        'score_threshold' : 0.5
    },
    'huge': {
        'model' : {
            'config' : join(mmdet_config_path, 'rtmdet-ins_x_8xb16-300e_coco.py'),
            'checkpoint' : join(checkpoint_dir, 'rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth'),
            'device': 'cuda:0'
        },
        'out_dir' : join(package_dir, 'rtmdet_out'),
        'score_threshold' : 0.5
    },
}