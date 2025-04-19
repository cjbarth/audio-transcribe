# Audio Transcribe

This project provides a script to transcribe audio or video files using either the `whisper` or `faster-whisper` library.

## Installation

1. Clone the repository or download the script.
2. Install the required dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script with the following command:

```bash
python audio-transcribe.py <path_to_audio_or_video_file>
```

### Options

- To use the `faster-whisper` library instead of `whisper`, add the `--use-faster` flag:

  ```bash
  python audio-transcribe.py <path_to_audio_or_video_file> --use-faster
  ```

### Saving Output to a File

To save the transcription output to a file, redirect the output:

```bash
python3 audio-transcribe.py <path_to_audio_or_video_file> > output.txt
```

For example:

```bash
python3 audio-transcribe.py example.mp3 > transcription.txt
```

This will save the transcription to `transcription.txt`.

## Notes

- Ensure you have a CUDA-compatible GPU if you want to leverage GPU acceleration.
- This script will only work with CUDA on Linux or WSL.
- The script will prompt for a file path if no file is provided as an argument.
