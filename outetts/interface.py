from .version.interface import InterfaceHF, InterfaceLLAMACPP, InterfaceEXL2
from .models.config import ModelConfig
from .models.info import Backend, InterfaceVersion
import os

def Interface(config: ModelConfig) -> InterfaceHF | InterfaceLLAMACPP | InterfaceEXL2:

    if config.backend == Backend.HF:
        return InterfaceHF(config)
    elif config.backend == Backend.LLAMACPP:
        return InterfaceLLAMACPP(config)
    elif config.backend == Backend.EXL2:
        return InterfaceEXL2(config)
    
    raise ValueError(f"Invalid backend: {config.backend} - must be one of {list(Backend)}")
