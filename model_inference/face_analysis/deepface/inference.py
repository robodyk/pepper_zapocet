from typing import Any
from deepface import DeepFace
from model_inference.face_analysis.deepface.configs import configs
import cv2

class Inferencer:
    def __init__(self, cfg_name, debug=False) -> None:
        cfg = configs[cfg_name]
        self.detector_backend = cfg['detector_backend']
        self.__call__('/home/martin/Projects/pepper_zapocet/model_inference/face_recognition/deepface/database/Krino/Krino.jpg')
    

    def __call__(self, image, image_name='out'):
        res = DeepFace.analyze(image,
                               detector_backend= self.detector_backend)
        return res[0]


if __name__ == '__main__':
    image = cv2.imread('/home/martin/Projects/pepper_zapocet/model_inference/face_recognition/test_images/tst1.jpg')
    model = Inferencer('default')
    model(image)