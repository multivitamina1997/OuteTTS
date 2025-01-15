import torch
from dataclasses import dataclass, field

@dataclass
class HFModelConfig:
    model_path: str = "OuteAI/OuteTTS-0.3-1B"
    tokenizer_path: str = None 
    verbose: bool = False
    device: str = None
    dtype: torch.dtype = None
    additional_model_config: dict = field(default_factory=dict)
    wavtokenizer_model_path: str = None
    max_seq_length: int = 4096

@dataclass
class GGUFModelConfig(HFModelConfig):
    n_gpu_layers: int = 0

@dataclass
class EXL2ModelConfig(HFModelConfig):
    pass