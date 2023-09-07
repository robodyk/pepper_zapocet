from typing import Any
from mmdet.apis import inference_detector, init_detector
from model_inference.detector.rtmdet.configs import configs
from mmdet.registry import VISUALIZERS
import numpy as np
import os
import cv2

class Inferencer:
    def __init__(self, config_name, debug=False) -> None:
        cfg = configs[config_name]
        self.model = init_detector(**cfg['model'])
        self.debug = debug
        self.score_threshold = cfg['score_threshold']
        self.out_dir = cfg['out_dir']
        if debug:
            vis_cfg = self.model.cfg.visualizer
            vis_cfg['save_dir'] = self.out_dir
            self.visualizer = VISUALIZERS.build(vis_cfg)
            self.visualizer.dataset_meta = self.model.dataset_meta

    def __call__(self, image, img_name='out') -> Any:
        result = inference_detector(self.model, image)

        if self.debug:
            self._save_vis(img_name, image, result)

        score_mask = result.pred_instances.scores > self.score_threshold
        ret = {
            'bboxes' : result.pred_instances.bboxes[score_mask].cpu().numpy(),
            'labels' : result.pred_instances.labels[score_mask].cpu().numpy(),
            'masks' : result.pred_instances.masks[score_mask].cpu().numpy(),
            'scores' : result.pred_instances.scores[score_mask].cpu().numpy(),
            'img_shape' : result.ori_shape,
            'detected' : np.any(score_mask.cpu().numpy())
        }

        return ret


    def _save_vis(self, img_name, image, result):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.visualizer.add_datasample(
            'result',
            image,
            data_sample=result,
            draw_gt=False,
            wait_time=0)
        
        im_fname = os.path.basename(img_name)

        self.visualizer.add_image(im_fname, self.visualizer.get_image())


if __name__ == '__main__':
    image = cv2.imread('/home/martin/Projects/pepper_zapocet/model_inference/detector/test_images/file_example_JPG_100kB.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model = Inferencer('default', True)
    res = model(image_rgb, 'pele3')
    print(res)