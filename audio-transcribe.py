import argparse
import os
import torch


def whisper(file):
    import whisper

    model_size = "turbo"
    model_size = "medium"
    model_size = "large"

    os.environ["TRITON_DISABLE_AUTOTUNE"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = whisper.load_model(model_size).to(device)
    result = model.transcribe(
        file,
        word_timestamps=True,
        compression_ratio_threshold=2.6,
        no_speech_threshold=0.6,
        temperature=(0.0,),
        fp16=False,
    )

    for segment in result["segments"]:
        print(
            "[%.2fs -> %.2fs] %s" % (segment["start"], segment["end"], segment["text"])
        )


def faster_whisper(file):
    from faster_whisper import WhisperModel

    model_size = "large-v3"
    model_size = "turbo"
    model_size = "medium"
    model_size = "small"

    model = WhisperModel(model_size, device="cuda", compute_type="float32")

    segments, info = model.transcribe(
        file,
        beam_size=5,
        patience=2,
        max_initial_timestamp=5.0,
    )

    print(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio/video with Whisper")
    parser.add_argument("file", nargs="?", help="Path to the input audio/video file")
    parser.add_argument(
        "--use-faster",
        action="store_true",
        help="Use faster_whisper instead of whisper",
    )
    args = parser.parse_args()

    if not args.file:
        args.file = input("Enter path to the audio/video file: ").strip()

    if args.use_faster:
        faster_whisper(args.file)
    else:
        whisper(args.file)
