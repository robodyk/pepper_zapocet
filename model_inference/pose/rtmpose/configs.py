from os.path import dirname, join

package_dir = dirname(__file__)
mmdet_path = '/home/martin/models/mmpose'
mmdet_config_path = join(mmdet_path, 'configs/rtmpose/')
checkpoint_dir = package_dir

configs = {
    'default': {
        'model' : {
            'config' : join(mmdet_config_path, 'rtmdet-ins_s_8xb32-300e_coco.py'),
            'checkpoint' : join(checkpoint_dir, 'rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth'),
            'device': 'cuda:0'
        },
        'out_dir' : join(package_dir, 'rtmdet_out'),
        'score_threshold' : 0.5
    },
}