from typing import Any
from deepface import DeepFace
from model_inference.face_recognition.deepface.configs import configs
import cv2
import os
import numpy as np
import logging
from deepface.commons import distance as dst

class Inferencer:

    def __init__(self, cfg_name, debug=False) -> None:
        cfg = configs[cfg_name]
        self.model_names = cfg['model_names']
        self.db_path = os.path.join(cfg['db_dir'], f'db_{self.model_names["embedding"]}.npz')
        self.match_threshold = dst.findThreshold(self.model_names['embedding'], 'cosine')
        self.embed_normalize('/home/martin/Projects/pepper_zapocet/model_inference/face_recognition/deepface/database/Krino/Krino.jpg')
        
        try:
            with np.load(self.db_path) as data:
                self.embeddings = data['embeddings']
                self.labels = data['labels']
        except FileNotFoundError:
            self.embeddings = np.empty((0, cfg['embedding_size']), dtype='float32')
            self.labels = np.array([], dtype='S8')


    def __call__(self, image, image_name='out'):
        embedding = self.embed_normalize(image)
        if self.embeddings.size == 0:
            self.embedding_cache = embedding
            return {'name': ''}
        cosine_sim = np.dot(self.embeddings, embedding)
        idx = np.argmax(cosine_sim)
        if cosine_sim[idx] > self.match_threshold: 
            name = self.labels[idx]
        else:
            name = ''
            self.embedding_cache = embedding
        return {'name' : name}


    def embed_normalize(self, image):
        res = DeepFace.represent(image,
                            model_name= self.model_names['embedding'],
                            detector_backend= self.model_names['detector_backend'])
        best_face_dict = max(res, key= lambda x: x['face_confidence'])
        x = np.array(best_face_dict['embedding'])
        return x / np.linalg.norm(x)


    def update_database(self, name, image, from_cache=False):
        if from_cache:
            if image: logging.warn("Unexpected arguments (from_cache => image is None)")
            embedding = self.embedding_cache
            self.embedding_cache = None
        elif not image:
            return
        else:
            embedding = self.embed_normalize(image)
        self.embeddings = np.vstack([self.embeddings, embedding[np.newaxis, :]])
        self.labels = np.append(self.labels, name)
        np.savez(self.db_path, embeddings=self.embeddings, labels=self.labels)
        

if __name__ == '__main__':
    image = cv2.imread('/home/martin/Projects/pepper_zapocet/model_inference/face_recognition/test_images/tst1.jpg')
    embedding = DeepFace.represent(image,
                            model_name= 'Facenet',
                            detector_backend= 'retinaface')

    model = Inferencer('default')
    print(model(image))