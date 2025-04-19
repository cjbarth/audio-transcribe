import argparse
import os
import torch
import time
import regex


def progress_callback(current_segments, processing_segment, left_segment, rate):
    for segment in current_segments:
        text = segment["text"]
        # Use regex to split text but keep punctuation with the previous segment
        parts = regex.split(r"(?<=[,\.!?;、。！？；])", text)
        for part in parts:
            if part.strip():
                # They could have resized the terminal during the `sleep`
                terminal_width = os.get_terminal_size().columns
                line = f"{rate*100:.1f}% {part}"
                if len(line) > terminal_width:
                    line = line[: terminal_width - 1] + "…"
                padded_line = line.ljust(terminal_width)
                print(padded_line, end="\r")
                time.sleep(0.3)


def write_segments_to_file(segments, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(
                "[%.2fs -> %.2fs] %s\n"
                % (segment["start"], segment["end"], segment["text"])
            )


def whisper(file, model_size, output):
    import whisper

    print(f"Loading Whisper model ({model_size})...")

    os.environ["TRITON_DISABLE_AUTOTUNE"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: No GPU detected. Using CPU for inference, which may be slow.")

    model = whisper.load_model(model_size).to(device)

    print("Starting transcription...")
    result = model.transcribe(
        file,
        word_timestamps=True,
        compression_ratio_threshold=2.6,
        no_speech_threshold=0.6,
        temperature=(0.0,),
        fp16=False,
        progress=progress_callback,  # Add progress callback
    )
    print("\nTranscription completed.")

    if output:
        write_segments_to_file(result["segments"], output)
        print(f"Transcription saved to {output}")
    else:
        for segment in result["segments"]:
            print(
                "[%.2fs -> %.2fs] %s"
                % (segment["start"], segment["end"], segment["text"])
            )


def faster_whisper(file, model_size, output):
    from faster_whisper import WhisperModel

    print(f"Loading Faster Whisper model ({model_size})...")

    model = WhisperModel(model_size, device="cuda", compute_type="float32")

    print("Starting transcription...")
    segments, info = model.transcribe(
        file,
        beam_size=5,
        patience=2,
        max_initial_timestamp=5.0,
        progress=progress_callback,  # Add progress callback
    )
    print("\nTranscription completed.")

    print(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )

    if output:
        with open(output, "w", encoding="utf-8") as f:
            for segment in segments:
                f.write(
                    "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
                )
        print(f"Transcription saved to {output}")
    else:
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio/video with Whisper")
    parser.add_argument("file", nargs="?", help="Path to the input audio/video file")
    parser.add_argument(
        "-f",
        "--faster",
        action="store_true",
        help="Use faster_whisper instead of whisper",
    )
    parser.add_argument(
        "-m",
        "--model-size",
        default="tiny.en",
        help="Set the model size (see https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to save the transcription output to a file",
    )
    args = parser.parse_args()

    if not args.file:
        args.file = input("Enter path to the audio/video file: ").strip()

    if args.faster:
        print("Using Faster Whisper for transcription.")
        faster_whisper(args.file, model_size=args.model_size, output=args.output)
    else:
        print("Using Whisper for transcription.")
        whisper(args.file, model_size=args.model_size, output=args.output)
