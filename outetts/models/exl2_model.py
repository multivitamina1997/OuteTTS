from loguru import logger
from .config import GenerationConfig

try:
    from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
    from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
    _EXL2_AVAILABLE = True
except:
    _EXL2_AVAILABLE = False

class EXL2Model:
    def __init__(
            self,
            model_path: str,
            max_seq_length: int,
            additional_model_config: dict = {},
    ) -> None:

        if not _EXL2_AVAILABLE:
            raise ImportError(
                "exllamav2 python module not found."
                "To use the EXL2 model you must install exllamav2 manually."
            )

        config = ExLlamaV2Config(model_path)
        config.arch_compat_overrides()
        self.model = ExLlamaV2(config)
        self.cache = ExLlamaV2Cache(self.model, max_seq_len=config.max_seq_len, lazy=True)
        self.model.load_autosplit(self.cache, progress=True)
        self.tokenizer = ExLlamaV2Tokenizer(config)

    def generate(self, input_ids: str, config: GenerationConfig, additional_dynamic_generator_config: dict, stream: bool = False) -> list[int]:
        generator = ExLlamaV2DynamicGenerator(
            model = self.model,
            cache = self.cache,
            tokenizer = self.tokenizer,
            **additional_dynamic_generator_config
        )
        if stream:
            raise NotImplementedError("Stream generation is not supported for EXL2 models.")

        gen_settings = ExLlamaV2Sampler.Settings(
                    token_repetition_penalty=config.repetition_penalty,
                    temperature=config.temperature,
                    **config.additional_gen_config
                ),

        input_size = self.tokenizer.encode(input_ids).size()[-1]

        output = generator.generate(
            prompt = input_ids,
            max_new_tokens = config.max_length,
            add_bos = False,
            decode_special_tokens=True,
            gen_settings = gen_settings,
            stop_conditions=[self.tokenizer.eos_token_id]
        )

        return self.tokenizer.encode(output).flatten().tolist()[input_size:]
