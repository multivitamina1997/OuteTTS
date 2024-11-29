from dataclasses import dataclass, field
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from llama_cpp import Llama, llama_token_is_eog
except ImportError:
    _GGUF_AVAILABLE = False
else:
    _GGUF_AVAILABLE = True

try:
    from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
    from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
except ImportError:
    _EXL2_AVAILABLE = False
else:
    _EXL2_AVAILABLE = True

@dataclass
class GenerationConfig:
    temperature: float = 0.1
    repetition_penalty: float = 1.1
    max_length: int = 4096
    additional_gen_config: dict = field(default_factory=lambda: {})

class HFModel:
    def __init__(
        self,
        model_path: str,
        device: str = None,
        dtype: torch.dtype = None,
        additional_model_config: dict = {}
    ) -> None:
        self.device = torch.device(
            device if device is not None 
            else "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            **additional_model_config
        ).to(device)

    def generate(self, input_ids: torch.Tensor, config: GenerationConfig) -> list[int]:
        return self.model.generate(
            input_ids,
            max_length=config.max_length,
            temperature=config.temperature,
            repetition_penalty=config.repetition_penalty,
            do_sample=True,
            **config.additional_gen_config,
        )[0].tolist()

class GGUFModel:
    def __init__(
            self,
            model_path: str,
            n_gpu_layers: int = 0,
            max_length: int = 4096,
            additional_model_config: dict = {}
    ) -> None:
        
        if not _GGUF_AVAILABLE:
            raise ImportError(
                "llama_cpp python module not found."
                "To use the GGUF model you must install llama cpp python manually."
            )

        additional_model_config["n_ctx"] = max_length
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            **additional_model_config
        )
    
    def generate(self, input_ids: list[int], config: GenerationConfig) -> list[int]:
        tokens = []
        for token in self.model.generate(
            input_ids,
            temp=config.temperature,
            repeat_penalty=config.repetition_penalty,
            **config.additional_gen_config,
        ):
            tokens.append(token)
            if (llama_token_is_eog(self.model._model.model, token) or 
                len(tokens) >= config.max_length):
                break

        return tokens

class EXL2Model:
    def __init__(
            self,
            model_path: str,
            additional_model_config: dict = {}
    ) -> None:
        
        if not _EXL2_AVAILABLE:
            raise ImportError(
                "exllamav2 python module not found."
                "To use the GGUF model you must install exllamav2 manually."
            )
        config = ExLlamaV2Config(model_path)
        config.arch_compat_overrides()
        self.model = ExLlamaV2(config)
        # Room for improvement: Cache is hardcoded to qwen-2.5-0.5B max seq len
        self.cache = ExLlamaV2Cache(self.model, max_seq_len=32768, lazy=True)
        self.model.load_autosplit(self.cache, progress=True)
        self.tokenizer = ExLlamaV2Tokenizer(config)
    
    def generate(self, input_ids: list[int], config: GenerationConfig) -> list[int]:
        generator = ExLlamaV2DynamicGenerator(
            model = self.model,
            cache = self.cache,
            max_length: int = 4096,
            tokenizer = self.tokenizer,
            gen_settings = ExLlamaV2Sampler.Settings(token_repetition_penalty=config.repetition_penalty, temperature=config.temperature, **config.additional_gen_config)
        )
        job = ExLlamaV2DynamicJob(input_ids=torch.tensor([input_ids]), max_new_tokens=max_length)
        generator.enqueue(job)
        eos = False
        tokens = []
        while not eos:
            results = generator.iterate()
            # Only batches one at a time. Has room for improvement.
            for result in results:
                assert result["job"] == job
                if result["stage"] == "streaming":
                    tokens.append(int(result.get("token_ids", "")[0][0]))
                    if tokens[-1] == self.tokenizer.eos_token_id:
                        eos = True
        #print(tokens)
        return tokens
