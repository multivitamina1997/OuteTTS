
## Interface v1 Usage

**Support models: OuteTTS-0.2, OuteTTS 0.1**

### Quick Start: Basic Full Example

```python
import outetts

# Configure the model
model_config = outetts.HFModelConfig_v1(
    model_path="OuteAI/OuteTTS-0.2-500M",
    language="en",  # Supported languages in v0.2: en, zh, ja, ko
)

# Initialize the interface
interface = outetts.InterfaceHF(model_version="0.2", cfg=model_config)

# Print available default speakers
interface.print_default_speakers()

# Load a default speaker
speaker = interface.load_default_speaker(name="male_1")

# Generate speech
output = interface.generate(
    text="Speech synthesis is the artificial production of human speech.",
    temperature=0.1,
    repetition_penalty=1.1,
    max_length=4096,

    # Optional: Use a speaker profile for consistent voice characteristics
    # Without a speaker profile, the model will generate a voice with random characteristics
    speaker=speaker,
)

# Save the generated speech to a file
output.save("output.wav")

# Optional: Play the generated audio
# output.play()
```

### Backend-Specific Configuration

#### Hugging Face Transformers

```python
import outetts

model_config = outetts.HFModelConfig_v1(
    model_path="OuteAI/OuteTTS-0.2-500M",
    language="en",  # Supported languages in v0.2: en, zh, ja, ko
)

interface = outetts.InterfaceHF(model_version="0.2", cfg=model_config)
```

#### GGUF (llama-cpp-python)

```python
import outetts

model_config = outetts.GGUFModelConfig_v1(
    model_path="local/path/to/model.gguf",
    language="en", # Supported languages in v0.2: en, zh, ja, ko
    n_gpu_layers=0,
)

interface = outetts.InterfaceGGUF(model_version="0.2", cfg=model_config)
```

#### ExLlamaV2

```python
import outetts

model_config = outetts.EXL2ModelConfig_v1(
    model_path="local/path/to/model",
    language="en", # Supported languages in v0.2: en, zh, ja, ko
)

interface = outetts.InterfaceEXL2(model_version="0.2", cfg=model_config)
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
speaker = interface.load_default_speaker(name="male_1")
```

### Text-to-Speech Generation

The generation process is consistent across all backends.

```python
output = interface.generate(
    text="Speech synthesis is the artificial production of human speech.",
    temperature=0.1,
    repetition_penalty=1.1,
    max_length=4096,
    speaker=speaker, # Optional: speaker profile
)

output.save("output.wav")
# Optional: Play the audio
# output.play()
```

### Custom Backend Configuration

You can initialize custom backend configurations for specific needs.

#### Example with Flash Attention for Hugging Face Transformers

```python
model_config = outetts.HFModelConfig_v1(
    model_path="OuteAI/OuteTTS-0.2-500M",
    language="en",
    dtype=torch.bfloat16,
    additional_model_config={
        'attn_implementation': "flash_attention_2"
    }
)
```

### Node.js Quick Start

The JavaScript implementation follows the same patterns as the Python version, making it easy to switch between the two.

```javascript
import { HFModelConfig_v1, InterfaceHF } from "outetts";

// Configure the model
const model_config = new HFModelConfig_v1({
    model_path: "onnx-community/OuteTTS-0.2-500M",
    language: "en", // Supported languages in v0.2: en, zh, ja, ko
    dtype: "fp32", // Supported dtypes: fp32, q8, q4
});

// Initialize the interface
const tts_interface = await InterfaceHF({ model_version: "0.2", cfg: model_config });

// Print available default speakers
tts_interface.print_default_speakers();

// Load a default speaker
const speaker = tts_interface.load_default_speaker("male_1");

// Generate speech
const output = await tts_interface.generate({
    text: "Speech synthesis is the artificial production of human speech.",
    temperature: 0.1, // Lower temperature values may result in a more stable tone
    repetition_penalty: 1.1,
    max_length: 4096,

    // Optional: Use a speaker profile for consistent voice characteristics
    // Without a speaker profile, the model will generate a voice with random characteristics
    speaker,
});

// Save the synthesized speech to a file
output.save("output.wav");
```

For browser-based applications, check out the example implementation: https://github.com/huggingface/transformers.js-examples/tree/main/text-to-speech-webgpu
