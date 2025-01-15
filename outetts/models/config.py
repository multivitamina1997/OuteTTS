from dataclasses import dataclass, field

@dataclass
class GenerationConfig:
    text: str = None
    voice_characteristics: str = None
    speaker: dict = None

    temperature: float = 0.1
    repetition_penalty: float = 1.1
    max_length: int = 4096
    additional_gen_config: dict = field(default_factory=lambda: {})
    additional_dynamic_generator_config: dict = field(default_factory=lambda: {})

