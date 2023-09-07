from os.path import dirname, join

package_dir = dirname(__file__)

configs = {
    'default':{
        'model_names': {'embedding': 'Facenet', 'detector_backend' : 'retinaface'},
        'db_dir': join(package_dir, 'database'),
        'embedding_size': 128,
    }
}