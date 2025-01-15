import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import GenerationConfig

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
        self.dtype = dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            **additional_model_config
        ).to(self.device)

    def generate(self, input_ids: torch.Tensor, config: GenerationConfig, stream: bool = False) -> list[int]:
        if stream:
            raise NotImplementedError("Stream generation is not supported for HF models.")

        return self.model.generate(
            input_ids,
            max_length=config.max_length,
            temperature=config.temperature,
            repetition_penalty=config.repetition_penalty,
            do_sample=True,
            **config.additional_gen_config,
        )[0].tolist()