from .inference_api import SingleImageAlphaPose
from .configs import configs
from alphapose.utils.config import update_config
import os
import cv2
import base64
import numpy as np
import torch

class Inferencer:
    def __init__(self, cfg_name, debug=False) -> None:
        args = configs[cfg_name]
        args.gpus = [int(args.gpus[0])] if torch.cuda.device_count() >= 1 else [-1]
        args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
        args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

        cfg = update_config(args.model_cfg)

        self.model = SingleImageAlphaPose(args, cfg)

        self.json_format = args.format
        self.json_for_eval = args.eval
        self.debug = debug
        self.out_path = args.outputpath

    def __call__(self, image, img_name='out'):


        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose = self.model.process(img_name, image)

        if pose is None:
            return {'detected' : False}

        # If debug, save the results to folder
        if self.debug:
            vis_path = os.path.join(self.out_path, 'vis')
            if not os.path.exists(vis_path):
                os.mkdir(vis_path)
            img = self.model.getImg()     # or you can just use: img = cv2.imread(image)
            img = self.model.vis(img, pose)   # visulize the pose result
            im_fname = os.path.basename(img_name)
            if not im_fname.endswith('.jpg'):
                im_fname += '.jpg'
            cv2.imwrite(os.path.join(vis_path, im_fname), img)

            # Write the result to json:
            self.model.writeJson([pose], self.out_path, form=self.json_format, for_eval=self.json_for_eval)

        best_pose = max(pose['result'], key=lambda x: x['proposal_score'].item())
        result = {'keypoints' : best_pose['keypoints'].cpu().numpy(),
                  'kp_score' : best_pose['kp_score'].cpu().numpy(),
                  'bbox' : best_pose['bbox'],
                  'detected' : True}

        return result

        results_score = []
        for res in json_results:
            results_score.append(res["score"])

        max_score = max(results_score)
        max_index = results_score.index(max_score)
        #print(max_index)

        new_results = []
        for res in json_results:
            if res["score"] == max_score:
                new_results.append(res)
    

