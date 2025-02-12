from loguru import logger
from .config import GenerationConfig
from tqdm import tqdm
from packaging import version

try:
    from llama_cpp import Llama, llama_token_is_eog
    from llama_cpp import __version__ as llama_cpp_version
    _GGUF_AVAILABLE = True
except:
    llama_cpp_version = "0.0.0"
    _GGUF_AVAILABLE = False

CURRENT_VERSION = version.parse(llama_cpp_version)
VERSION_0_3_7 = version.parse("0.3.7")

class GGUFModel:
    def __init__(
            self,
            model_path: str,
            n_gpu_layers: int = 0,
            max_seq_length: int = 4096,
            additional_model_config: dict = {}
    ) -> None:

        if not _GGUF_AVAILABLE:
            raise ImportError(
                "llama_cpp python module not found."
                "To use the GGUF model you must install llama cpp python manually."
            )

        additional_model_config["n_ctx"] = max_seq_length
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            **additional_model_config
        )

    def is_eog(self):
        if CURRENT_VERSION >= VERSION_0_3_7:
            return self.model._model.vocab
        else:
            return self.model._model.model

    def generate(self, input_ids: list[int], config: GenerationConfig, stream: bool = False):
        if stream:
            return self._generate_stream(input_ids, config)
        return self._generate(input_ids, config)

    def _generate_stream(self, input_ids: list[int], config: GenerationConfig):
        size = 0
        input_size = len(input_ids)
        for token in self.model.generate(
            input_ids,
            temp=config.temperature,
            repeat_penalty=config.repetition_penalty,
            **config.additional_gen_config,
        ):
            yield token
            size += 1
            if (llama_token_is_eog(self.is_eog(), token) or 
                size + input_size >= config.max_length):
                break

    def _generate(self, input_ids: list[int], config: GenerationConfig) -> list:
        input_size = len(input_ids)
        tokens = []
        gen = tqdm(self.model.generate(
            input_ids,
            temp=config.temperature,
            repeat_penalty=config.repetition_penalty,
            **config.additional_gen_config,
        ))
        for token in gen:
            tokens.append(token)
            if (llama_token_is_eog(self.is_eog(), token) or 
                len(tokens) + input_size >= config.max_length):
                break
            gen.set_postfix({"tokens": input_size + len(tokens), "max tokens": config.max_length})

        return tokens
