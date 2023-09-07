import whisper

class Inferencer:

    def __init__(self, cfg_name, debug=False) -> None:
        self.model = whisper.load_model(cfg_name)

    def __call__(self, audio_file, _=None):
        result = self.model.transcribe(audio_file)
        return result['text']