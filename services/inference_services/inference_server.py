from services.service_server import ServiceServer
from os.path import dirname, join
from model_inference import pose
import importlib
import numpy as np
import cv2
import base64
import time

class InferenceServer(ServiceServer):
    def __init__(self, model_dict, cfg_dict, debug=False) -> None:
        self.models = {}
        for mfunc in ['pose', 'detector', 'face_analysis', 'face_recognition', 'speech']:
            module_path = f'model_inference.{mfunc}.{model_dict[mfunc]}.inference'
            model_module = importlib.import_module(module_path)
            self.models[mfunc] = model_module.Inferencer(cfg_dict[mfunc], debug)
        super().__init__(self.callback)
    
    
    
    def callback(self, received_data):
        
        request_type = received_data['request_type']
        if request_type == 'face_database':
            self.models['face_recognition'].update_database(
                name= received_data['name'],
                image= received_data['image_as_txt'],
                from_cache= received_data['from_cache']
            )
            return {'name' : received_data['name']}

        if request_type == 'speech':
            return self.models[request_type](received_data['filename'])
        
        img_name = received_data["image_name"]   # final name of the image
        # Decode image from string
        image_as_txt = received_data["image_as_txt"]
        jpg_original = base64.b64decode(image_as_txt)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        image = cv2.imdecode(jpg_as_np, flags=1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return self.models[request_type](image, img_name)

if __name__ == '__main__':
    model_dict = {'pose' : 'alphapose', 'detector' : 'rtmdet', 'face_analysis' : 'deepface', 'face_recognition' : 'deepface', 'speech' : 'whisper'}
    cfg_dict = {'pose' : 'default', 'detector' : 'huge', 'face_analysis' : 'default', 'face_recognition' : 'default', 'speech' : 'base.en'}
    server = InferenceServer(model_dict, cfg_dict, debug=True)
    try:
        while(True):
            time.sleep(1)
    except KeyboardInterrupt as e:
        print(e)