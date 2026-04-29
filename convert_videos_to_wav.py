import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT_DIRS = [
    PROJECT_ROOT / "data" / "thai_video",
    PROJECT_ROOT / "data" / "eng_video",
    PROJECT_ROOT / "data" / "video",
]

OUTPUT_MAP = {
    "thai_video": PROJECT_ROOT / "data" / "thai_wav",
    "eng_video": PROJECT_ROOT / "data" / "eng_wav",
    "video": PROJECT_ROOT / "data" / "wav",
}

def find_ffmpeg():
    for cmd in ["ffmpeg", "ffmpeg.exe"]:
        try:
            subprocess.run([cmd, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return cmd
        except Exception:
            continue
    return None


def convert_video_to_wav(input_path: Path, output_path: Path, ffmpeg_cmd: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_cmd,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def gather_video_files(input_dirs):
    files = []
    for input_dir in input_dirs:
        if not input_dir.exists():
            continue
        for path in input_dir.rglob("*.mp4"):
            if path.is_file():
                files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Convert saved gesture video files to WAV audio files." 
                    "By default it scans data/thai_video, data/eng_video, and data/video.",
    )
    parser.add_argument(
        "--input",
        nargs="*",
        default=[str(p) for p in DEFAULT_INPUT_DIRS],
        help="Input video directories to scan.",
    )
    parser.add_argument(
        "--output-base",
        default=str(PROJECT_ROOT / "data"),
        help="Base output directory for WAV files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing WAV files.",
    )
    args = parser.parse_args()

    ffmpeg_cmd = find_ffmpeg()
    if ffmpeg_cmd is None:
        print("Error: ffmpeg is required but was not found on PATH.")
        print("Please install ffmpeg and make sure it is available in your shell.")
        sys.exit(1)

    input_dirs = [Path(path).resolve() for path in args.input]
    output_base = Path(args.output_base).resolve()
    video_files = gather_video_files(input_dirs)

    if not video_files:
        print("No video files found in the configured input folders.")
        print("Checked:")
        for input_dir in input_dirs:
            print(f" - {input_dir}")
        sys.exit(0)

    print(f"Found {len(video_files)} video file(s) to convert.")
    converted = 0
    skipped = 0

    for video_path in sorted(video_files):
        rel = None
        for input_dir in input_dirs:
            try:
                rel = video_path.relative_to(input_dir)
                input_dir_name = input_dir.name
                break
            except ValueError:
                continue

        if rel is None:
            output_root = output_base / "wav"
        else:
            output_root = OUTPUT_MAP.get(input_dir_name, output_base / (input_dir_name + "_wav"))

        output_path = output_root / rel.with_suffix(".wav")

        if output_path.exists() and not args.force:
            skipped += 1
            print(f"Skip existing: {output_path}")
            continue

        try:
            convert_video_to_wav(video_path, output_path, ffmpeg_cmd)
            print(f"Converted: {video_path} -> {output_path}")
            converted += 1
        except subprocess.CalledProcessError as exc:
            print(f"Failed: {video_path} ({exc})")

    print(f"\nFinished: {converted} converted, {skipped} skipped.")
    print(f"WAV files saved under: {output_base}")


if __name__ == "__main__":
    main()
