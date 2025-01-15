
## Interface v2 Usage

**Support models: OuteTTS-0.3**

### Quick Start: Basic Full Example

```python
import outetts

# Configure the model
model_config = outetts.HFModelConfig_v2(
    model_path="OuteAI/OuteTTS-0.3-1B",
)
# Initialize the interface
interface = outetts.InterfaceHF(model_version="0.3", cfg=model_config)

# You can create a speaker profile for voice cloning, which is compatible across all backends.
# speaker = interface.create_speaker(audio_path="path/to/audio/file.wav")
# interface.save_speaker(speaker, "speaker.json")
# speaker = interface.load_speaker("speaker.json")

# Print available default speakers
interface.print_default_speakers()
# Load a default speaker
speaker = interface.load_default_speaker(name="en_male_1")

# Generate speech
gen_cfg = outetts.GenerationConfig(
    text="Speech synthesis is the artificial production of human speech.",
    temperature=0.1,
    repetition_penalty=1.1,
    max_length=4096,
    speaker=speaker,
    # voice_characteristics="upbeat enthusiasm, friendliness, clarity, professionalism, and trustworthiness"
)
output = interface.generate(config=gen_cfg)

# Save the generated speech to a file
output.save("output.wav")
```

### Backend-Specific Configuration

```python
import outetts

model_config = outetts.HFModelConfig_v2(
    model_path="OuteAI/OuteTTS-0.3-1B",
)

interface = outetts.InterfaceHF(model_version="0.3", cfg=model_config)
```

#### GGUF (llama-cpp-python)

```python
import outetts

model_config = outetts.GGUFModelConfig_v2(
    model_path="local/path/to/model.gguf",
    n_gpu_layers=0,
)

interface = outetts.InterfaceGGUF(model_version="0.3", cfg=model_config)
```

#### ExLlamaV2

```python
import outetts

model_config = outetts.EXL2ModelConfig_v2(
    model_path="local/path/to/model",
)

interface = outetts.InterfaceEXL2(model_version="0.3", cfg=model_config)
```

### Speaker Creation and Management

#### Creating a Speaker

You can create a speaker profile for voice cloning, which is compatible across all backends.

```python
speaker = interface.create_speaker(
    audio_path="path/to/audio/file.wav",

    # If transcript is not provided, it will be automatically transcribed using Whisper
    transcript=None,            # Set to None to use Whisper for transcription

    whisper_model="turbo",      # Optional: specify Whisper model (default: "turbo")
    whisper_device=None,        # Optional: specify device for Whisper (default: None)
)
```
#### Saving and Loading Speaker Profiles

Speaker profiles can be saved and loaded across all supported backends.

```python
# Save speaker profile
interface.save_speaker(speaker, "speaker.json")

# Load speaker profile
speaker = interface.load_speaker("speaker.json")
```

#### Default Speaker Initialization

OuteTTS includes a set of default speaker profiles. Use them directly:

```python
# Print available default speakers
interface.print_default_speakers()
# Load a default speaker
speaker = interface.load_default_speaker(name="en_male_1")
```

### Text-to-Speech Generation

The generation process is consistent across all backends.

```python
gen_cfg = outetts.GenerationConfig(
    text="Speech synthesis is the artificial production of human speech.",
    temperature=0.1,
    repetition_penalty=1.1,
    max_length=4096,
    speaker=speaker,
)
output = interface.generate(config=gen_cfg)

output.save("output.wav")
# Optional: Play the audio
# output.play(backend="...") # backend: str -> "sounddevice", "pygame"
```

### Custom Backend Configuration

You can initialize custom backend configurations for specific needs.

#### Example with Flash Attention for Hugging Face Transformers

```python
model_config = outetts.HFModelConfig_v2(
    model_path="OuteAI/OuteTTS-0.3-1B",
    dtype=torch.bfloat16,
    additional_model_config={
        'attn_implementation': "flash_attention_2"
    }
)
```