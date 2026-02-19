# Voice Reference Audio

Place voice reference audio files here for TTS voice cloning with CosyVoice 3.0.

## Requirements

| Property | Requirement |
|----------|-------------|
| Duration | 3-10 seconds |
| Format | WAV (recommended), MP3, OGG, or M4A |
| Sample Rate | 16 kHz or higher |
| Content | Clear speech, single speaker, minimal background noise |
| Language | Arabic preferred (matches default pipeline language) |

## How It Works

1. **Startup scan**: On launch, the TTS engine scans this directory and registers every audio file as an available voice. The filename (without extension) becomes the `voice_id`.

2. **API cloning**: Voices can also be added at runtime via `POST /api/v1/voice-clone`. Cloned voices are saved here automatically with an ID like `voice_a1b2c3d4.wav`.

## Example

```
voices/
├── salem.wav              # voice_id = "salem"
├── fatima.wav             # voice_id = "fatima"
├── voice_a1b2c3d4.wav     # voice_id = "voice_a1b2c3d4" (API-cloned)
└── README.md
```

## Usage

Reference a voice by ID in API calls:

```json
{
  "text": "Hello",
  "voice_id": "salem"
}
```

If no `voice_id` is provided, the TTS engine uses its default instruct-mode synthesis.

## Tips

- Record in a quiet environment with minimal echo
- Use a consistent speaking pace and clear pronunciation
- Avoid music, background noise, or multiple speakers
- 5-7 seconds of natural speech gives the best cloning results
