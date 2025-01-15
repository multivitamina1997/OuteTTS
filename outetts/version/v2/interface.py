import torch
import torchaudio
from dataclasses import dataclass, field
from loguru import logger
import os
import json
import threading
import queue
import re
import polars as pl
from pathlib import Path

from ...wav_tokenizer.audio_codec import AudioCodec
from .prompt_processor import PromptProcessor
from ...models.hf_model import HFModel
from ...models.gguf_model import GGUFModel
from ...models.exl2_model import EXL2Model
from ...models.config import GenerationConfig
from ...whisper import transcribe
from .config import HFModelConfig, GGUFModelConfig, EXL2ModelConfig
from ..playback import ModelOutput
from .alignment import CTCForcedAlignment

_BASE_DIR = os.path.dirname(__file__)
_DEFAULT_SPEAKERS_DIR = os.path.join(_BASE_DIR, "default_speakers/parquet/speakers.parquet")

if Path(_DEFAULT_SPEAKERS_DIR).exists():
    _DEFAULT_SPEAKERS = {i['speaker']: i for i in pl.read_parquet(_DEFAULT_SPEAKERS_DIR).to_dicts()}
else:
    _DEFAULT_SPEAKERS = None

class InterfaceHF:
    def __init__(
        self,
        config: HFModelConfig
    ) -> None:
        self.config = config
        self.verbose = config.verbose
        self.audio_codec = AudioCodec(config.device, config.wavtokenizer_model_path)
        self.prompt_processor = PromptProcessor(config.tokenizer_path)
        self.model = self.get_model()
        
    def get_model(self):
        return HFModel(
            self.config.model_path, 
            self.config.device, 
            self.config.dtype, 
            self.config.additional_model_config
        )

    def _prepare_prompt(self, prompt: str):
        return self.prompt_processor.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

    def check_input_prompt_size(self, text: str):
        input_ids = self.prompt_processor.tokenizer.encode(text, add_special_tokens=False)
        if len(input_ids) >= 1500:
            logger.warning(
            f"Input prompt size exceeds 1500 tokens (current size: {len(input_ids)} tokens). "
            "This may reduce the effective context window for the model and impact generation accuracy.\n"
            "To achieve the best results, aim for the generation window to fall within a 1-30 second range.\n"
            "If you experience issues with the model's output, consider using a shorter speaker reference or reducing the length of your input text."
        )

    def prepare_prompt(self, text: str, voice_characteristics: str = None, speaker: dict = None):
        prompt = self.prompt_processor.get_completion_prompt(text, voice_characteristics, speaker)
        self.check_input_prompt_size(prompt)
        return self._prepare_prompt(prompt)

    def get_audio(self, tokens):
        output = self.prompt_processor.extract_audio_from_tokens(tokens)
        if not output:
            logger.warning("No audio tokens found in the output")
            return None

        return self.audio_codec.decode(
            torch.tensor([[output]], dtype=torch.int64).to(self.audio_codec.device)
        )

    def create_speaker(
            self,
            audio_path: str,
            transcript: str = None,
            whisper_model: str = "turbo",
            whisper_device = None,
            free_ctc_model: bool = True
        ):
        if transcript is None:
            logger.info("Transcription not provided, transcribing audio with whisper.")
            transcript = transcribe.transcribe_once(
                audio_path=audio_path,
                model=whisper_model,
                device=whisper_device
            )
        if not transcript: 
            raise ValueError("Transcript text is empty")

        if not hasattr(self, 'ctc') or not isinstance(self.ctc, CTCForcedAlignment):
            self.ctc = CTCForcedAlignment(self.config.device)

        words = self.ctc.align(audio_path, transcript)
        full_codes = self.audio_codec.encode(
            self.audio_codec.convert_audio_tensor(
                audio=torch.cat([i["audio"] for i in words], dim=1),
                sr=self.ctc.sample_rate
            ).to(self.audio_codec.device)
        ).tolist()

        data = []
        start = 0
        for i in words:
            end = int(round((i["x1"] / self.ctc.sample_rate) * 75))
            word_tokens = full_codes[0][0][start:end]
            start = end
            if not word_tokens:
                word_tokens = [1]

            data.append({
                "word": i["word"],
                "duration": round(len(word_tokens) / 75, 2),
                "codes": word_tokens
            })

        if free_ctc_model:
            self.ctc.free()
            del self.ctc

        return {
            "text": transcript,
            "words": data,
        }

    def save_speaker(self, speaker: dict, path: str):
        with open(path, "w") as f: json.dump(speaker, f, indent=2)

    def load_speaker(self, path: str):
        with open(path, "r") as f: return json.load(f)

    def print_default_speakers(self):
        if _DEFAULT_SPEAKERS is None:
            logger.error(f"There was an issue loading default speakers.")
        else:
            logger.info(f"Available default speakers v2: {list(_DEFAULT_SPEAKERS.keys())}")
    
    def load_default_speaker(self, name: str):
        name = name.lower().strip()
        if _DEFAULT_SPEAKERS is None:
            raise ValueError(f"There was an issue loading default speakers.")

        if name not in _DEFAULT_SPEAKERS:
            raise ValueError(f"Speaker {name} not found for language {list(_DEFAULT_SPEAKERS.keys())}")

        return _DEFAULT_SPEAKERS.get(name, {})

    def check_generation_max_length(self, max_length):
        if max_length is None:
            raise ValueError("max_length must be specified.")
        if max_length > self.config.max_seq_length:
            raise ValueError(f"Requested max_length ({max_length}) exceeds the current max_seq_length ({self.config.max_seq_length}).")

    def _generate(self, input_ids, config: GenerationConfig):
        output = self.model.generate(
            input_ids=input_ids,
            config=config
        )
        return output[input_ids.size()[-1]:]
    
    def generate(self, config: GenerationConfig) -> ModelOutput:
        self.check_generation_max_length(config.max_length)
        if config.text is None:
            raise ValueError("text can not be empty!")
        if config.voice_characteristics is not None:
            logger.warning("Voice characteristics is an experimental feature that may not have any effect on the output. The model might process these settings differently than expected.")
        input_ids = self.prepare_prompt(config.text, config.voice_characteristics, config.speaker)
        output = self._generate(input_ids, config)
        audio = self.get_audio(output)
        return ModelOutput(audio, self.audio_codec.sr)

class InterfaceGGUF(InterfaceHF):
    def __init__(self, config: GGUFModelConfig) -> None:
        super().__init__(config)
        self.config = config

    def get_model(self):
        return GGUFModel(
            model_path=self.config.model_path,
            n_gpu_layers=self.config.n_gpu_layers,
            max_seq_length=self.config.max_seq_length,
            additional_model_config=self.config.additional_model_config
        )

    def _generate(self, input_ids, config):
        return self.model.generate(
            input_ids=input_ids,
            config=config,
        )

    def _prepare_prompt(self, prompt: str):
        return self.prompt_processor.tokenizer.encode(prompt, add_special_tokens=False)

class InterfaceEXL2(InterfaceHF):
    def __init__(self, config: EXL2ModelConfig) -> None:
        super().__init__(config)
        self.config = config

    def get_model(self):
        return EXL2Model(
            model_path=self.config.model_path,
            max_seq_length=self.config.max_seq_length,
            additional_model_config=self.config.additional_model_config,
        )

    def _prepare_prompt(self, prompt: str):
        return prompt
    
    def _generate(self, input_ids, config):
        return self.model.generate(
            input_ids=input_ids,
            config=config,
            additional_dynamic_generator_config=config.additional_dynamic_generator_config
        )



