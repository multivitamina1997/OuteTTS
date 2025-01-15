from loguru import logger
from .config import GenerationConfig
from tqdm import tqdm

try:
    from llama_cpp import Llama, llama_token_is_eog
    _GGUF_AVAILABLE = True
except:
    _GGUF_AVAILABLE = False

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
            if (llama_token_is_eog(self.model._model.model, token) or 
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
            if (llama_token_is_eog(self.model._model.model, token) or 
                len(tokens) + input_size >= config.max_length):
                break
            gen.set_postfix({"tokens": input_size + len(tokens), "max tokens": config.max_length})

        return tokens