import torch
import torchaudio
from loguru import logger
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

try: 
    import sounddevice as sd
except: 
    logger.warning("[playback] Failed to import sounddevice.")
try: 
    import pygame
except: 
    logger.warning("[playback] Failed to pygame sounddevice.")

class ModelOutput:
    def __init__(self, audio: torch.Tensor, og_sr: int):
        self.sr = 44100
        self.audio = self.resample(audio, og_sr, self.sr)

    def resample(self, audio: torch.Tensor, og_sr: int, to_sr: int):
        resampler = torchaudio.transforms.Resample(orig_freq=og_sr, new_freq=to_sr).to(audio.device)
        return resampler(audio)

    def save(self, path: str):
        if self.audio is None:
            logger.warning("Audio is empty, skipping save.")
            return
        torchaudio.save(path, self.audio.cpu(), sample_rate=self.sr, encoding='PCM_S', bits_per_sample=16)

    def _sounddevice(self):
        try:
            sd.play(self.audio[0].cpu().numpy(), self.sr)
            sd.wait()
        except Exception as e:
            logger.error(e)

    def _pygame(self):
        try:
            pygame.mixer.init(frequency=self.sr, channels=2)
            audio_data = self.audio[0].cpu().numpy()
            sound_array = (audio_data * 32767).astype('int16')
            if sound_array.ndim == 1:
                sound_array = np.expand_dims(sound_array, axis=1)
                sound_array = np.repeat(sound_array, 2, axis=1)
            sound = pygame.sndarray.make_sound(sound_array)
            sound.play()
            pygame.time.wait(int(sound.get_length() * 1000))
            pygame.mixer.quit()
        except Exception as e:
            logger.error(e)

    def _invalid_backend(self):
        logger.warning(f"Invalid backend selected!")

    def play(self, backend: str):
        """
        backend: str -> "sounddevice", "pygame"
        """
        if self.audio is None:
            logger.warning("Audio is empty, skipping playback.")
            return

        backends = {
            "sounddevice": self._sounddevice,
            "pygame": self._pygame
        }

        backend = backend.lower()
        backends.get(backend, self._invalid_backend)()
