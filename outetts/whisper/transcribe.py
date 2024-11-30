import whisper
from loguru import logger

def transcribe_once(audio_path: str, model: str = "turbo", device = None):
    logger.info(f"Loading model {model}")
    model = whisper.load_model("turbo", device=device) 
    logger.info(f"Transcribing {audio_path}")
    text = model.transcribe(audio_path)["text"]
    logger.success(f"Transcription: {text}")
    return text
