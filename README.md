# TODO: Clean this mess up

## Structure

- **Inference Server**: 
  - Path: `services/inference_services/inference_server.py`
  - *Note*: Should run this from an external script.

- **Model Inference**: 
  - Description: Inference abstraction for used models and their configurations.

## Python 2 Dependencies

- Install `pepper-controller`
- Install `zmq`
- Install `cloudpickle`

**pepper_people**:
- Contains the RPC classes and abstraction wrappers.

## Demo Scripts

- Located in `Pepper-Controller`:
  - `detDemo.py`
  - `faceDemo.py`

## Installation

### The model Libraries (Python3):

- Install `mmdet`
- Install `alphapose`
- Install `deepface`
- Install `whisper`

### Additional Python 3 Dependencies:

- `cloudpickle`
- `easydict`