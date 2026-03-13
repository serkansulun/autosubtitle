#!/usr/bin/env python3
"""
autoasr_whisper_generate.py

- Uses Hugging Face WhisperForConditionalGeneration.generate(..., return_timestamps=True)
    to get segment-level timestamps.
- Manually chunks audio (chunk_length_s, chunk_overlap_s) to avoid passing
    unsupported kwargs into model.generate().
- Keeps your parser defaults exactly.
- No error handling (per request). Timestamps are required; if Whisper doesn't
    predict an explicit end timestamp for a token, the script computes sensible
    fallback end times (next segment start or chunk end).
- Outputs .sub files (SRT-style) with the same basename as each .opus file.
- Prepends a copy of the first 30s of audio, drops segments starting in that window,
    then shifts timestamps back by 30s.
- Limits each subtitle to max_sub_duration seconds.

Requirements:
        pip install torch torchaudio transformers tqdm
        ffmpeg and ffprobe on PATH

Usage:
        python3 autoasr_whisper_generate.py
        (uses defaults: input_dir=data, output_dir=output, model=openai/whisper-large-v3)
"""

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import shutil

# --------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------
def ffmpeg_to_wav(input_path: Path, out_wav: Path):
    subprocess.run([
        "ffmpeg", "-y", "-i", str(input_path), "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(out_wav)
    ])

def ffmpeg_audio_to_black_video(input_audio: Path, out_video: Path, duration_s: float):
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=1280x720:d={duration_s}:r=30",
        "-i", str(input_audio),
        "-vf", "scale=1280:720",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-r", "30",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(out_video),
    ])

def ffprobe_duration(path: Path) -> float:
    p = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], stdout=subprocess.PIPE)
    return float(p.stdout.decode("utf-8").strip())

def chunk_ranges(total_dur: float, chunk_length_s: float, chunk_overlap_s: float) -> List[Tuple[float, float]]:
    step = chunk_length_s - chunk_overlap_s
    starts = []
    start = 0.0
    while start < total_dur:
        length = min(chunk_length_s, total_dur - start)
        starts.append((start, length))
        start += step
    return starts

def seconds_to_srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hours = ms // (3600 * 1000)
    ms -= hours * 3600 * 1000
    minutes = ms // (60 * 1000)
    ms -= minutes * 60 * 1000
    secs = ms // 1000
    ms -= secs * 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

def write_srt(segments, out_path: Path):
    lines = []
    for i, seg in enumerate(segments, start=1):
        start_ts = seconds_to_srt_timestamp(seg["start"])
        end_ts = seconds_to_srt_timestamp(seg["end"])
        text = seg["text"].replace("\n", " ").strip()
        lines.append(f"{i}\n{start_ts} --> {end_ts}\n{text}\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# --------------------------------------------------------------------------------
# Model builder (loads processor + model; moves model to device in FP16)
# --------------------------------------------------------------------------------
def build_whisper(model_name: str, device: int):
    # load processor (feature_extractor + tokenizer)
    proc = WhisperProcessor.from_pretrained(model_name)
    # load model and move to cuda with FP16 dtype
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    model.to(device_str, dtype=torch.float16 if "cuda" in device_str else torch.float32)
    model.eval()
    return proc, model, device_str

# --------------------------------------------------------------------------------
# Core transcription using model.generate(return_timestamps=True)
# --------------------------------------------------------------------------------
def transcribe_opus_file(proc: WhisperProcessor, model: WhisperForConditionalGeneration, device_str: str,
                         input_opus: Path, out_sub: Path,
                         batch_size: int, chunk_length_s: float, chunk_overlap_s: float,
                        max_sub_duration: float):

    td = Path('.tmp')
    td.mkdir(parents=True, exist_ok=True)

    # video_out = td / (input_opus.stem + ".mp4")
    # video_duration = ffprobe_duration(input_opus)
    # ffmpeg_audio_to_black_video(input_opus, video_out, video_duration)

    wav_full = td / (input_opus.stem + ".wav")
    ffmpeg_to_wav(input_opus, wav_full)

    wav_padded = td / (input_opus.stem + "_pad.wav")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(wav_full),
        "-filter_complex", "[0:a]atrim=0:30,asetpts=N/SR/TB[a0];[a0][0:a]concat=n=2:v=0:a=1[a]",
        "-map", "[a]",
        "-ac", "1", "-ar", "16000",
        str(wav_padded),
    ])

    total_dur = ffprobe_duration(wav_padded)

    ranges = chunk_ranges(total_dur, chunk_length_s, chunk_overlap_s)

    # Create chunk wavs
    chunk_files = []
    for idx, (start, length) in enumerate(ranges):
        chunk_wav = td / f"{input_opus.stem}_chunk{idx:04d}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(start), "-t", str(length),
            "-i", str(wav_padded), "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(chunk_wav)
        ])
        chunk_files.append((chunk_wav, start, length))

    # Process chunks in batches
    global_segments = []
    dtype = torch.float16 if "cuda" in device_str else torch.float32

    for i in tqdm(range(0, len(chunk_files), batch_size), desc=f"Chunks {input_opus.stem}", leave=False):
        batch = chunk_files[i:i+batch_size]

        # Load audio for each chunk into numpy/torch and prepare input_features
        inputs_list = []
        for (chunk_path, chunk_start, chunk_len) in batch:
            waveform, sr = torchaudio.load(str(chunk_path))  # [channels, frames]
            if waveform.ndim == 2:
                waveform = waveform.mean(dim=0)  # mix to mono
            audio_np = waveform.cpu().numpy()
            proc_inputs = proc(audio_np, sampling_rate=sr, return_tensors="pt")
            inputs_list.append((proc_inputs.input_features, chunk_start, chunk_len))

        # Generate per chunk (we do one-by-one generate inside batch loop to avoid
        # potentially incompatible batch-level shapes for Whisper generate timestamps)
        # (This still benefits from sending tensors to GPU)
        for input_features, chunk_start, chunk_len in inputs_list:
            # move to device & dtype
            input_features = input_features.to(device_str).to(dtype)
            with torch.inference_mode():
                gen_out = model.generate(
                    input_features,
                    max_length=448,
                    return_dict_in_generate=True,
                    return_timestamps=True,
                )

            # extract 'segments' from generation output (structure from HF generation_whisper)
            # gen_out may be a ModelOutput with attribute 'segments' or dict-like
            segments_obj = None
            if isinstance(gen_out, dict):
                segments_obj = gen_out.get("segments", None)
            else:
                segments_obj = getattr(gen_out, "segments", None)

            if segments_obj is None:
                # fallback: get plain text if available
                text = ""
                if isinstance(gen_out, dict) and "text" in gen_out:
                    text = gen_out["text"]
                elif hasattr(gen_out, "sequences"):
                    # decode sequences
                    text = proc.tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)[0]
                global_segments.append({"start": chunk_start, "end": chunk_start + chunk_len, "text": text})
                continue

            # segments_obj is typically a list with one element per batch item
            # We processed single item per generate call, so take first element
            segmented_output = segments_obj[0]

            # Collect start times and end time fallback if some 'end' missing.
            # Build segment-by-segment: for each segment, start = seg['start'],
            # end = next_segment.start (if next exists) else seg.get('end') or chunk_len
            # decode tokens with tokenizer
            seg_texts = []
            seg_starts = []

            for seg in segmented_output:
                # 'tokens' may be tensor; convert to list
                tokens = seg.get("tokens", None)
                if isinstance(tokens, torch.Tensor):
                    toks = tokens.cpu().tolist()
                else:
                    toks = tokens
                # decode
                text = proc.tokenizer.decode(toks, skip_special_tokens=True).strip() if toks is not None else seg.get("text", "").strip()
                seg_texts.append(text)
                start_val = seg.get("start", None)
                if isinstance(start_val, torch.Tensor):
                    start_val = float(start_val.cpu().item())
                seg_starts.append(float(start_val))

            # end_time fallback:
            last_seg = segmented_output[-1]
            last_end = last_seg.get("end", None)
            if isinstance(last_end, torch.Tensor):
                last_end = float(last_end.cpu().item())
            if last_end is None:
                last_end = chunk_len

            # now build per-segment start/end using next-start trick
            for idx_seg in range(len(seg_texts)):
                s = seg_starts[idx_seg]
                if idx_seg + 1 < len(seg_texts):
                    e = seg_starts[idx_seg + 1]
                else:
                    e = last_end
                # global timestamps
                global_segments.append({"start": s + chunk_start, "end": e + chunk_start, "text": seg_texts[idx_seg]})

    # drop segments that begin in the first 30 seconds, then shift back by 30s
    adjusted_segments = []
    for seg in global_segments:
        if seg["start"] < 30.0:
            continue
        new_start = seg["start"] - 30.0
        new_end = seg["end"] - 30.0
        if new_end < new_start:
            new_end = new_start
        if max_sub_duration is not None and max_sub_duration > 0:
            if new_end - new_start > max_sub_duration:
                new_end = new_start + max_sub_duration
        adjusted_segments.append({"start": new_start, "end": new_end, "text": seg["text"]})

    # write .sub (SRT-style)
    write_srt(adjusted_segments, out_sub)
    # Remove temporary directory
    shutil.rmtree(td, ignore_errors=True)

# --------------------------------------------------------------------------------
# Main with your exact defaults
# --------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--model", default="openai/whisper-large-v3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--chunk_length_s", type=int, default=30)
    parser.add_argument("--chunk_overlap_s", type=float, default=0.5)
    parser.add_argument("--max_sub_duration", type=float, default=5)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() == ".opus"])

    torch.backends.cudnn.benchmark = True

    proc, model, device_str = build_whisper(args.model, device=args.device)

    for ap in tqdm(audio_paths, desc="Files"):
        out_sub = out_dir / (ap.stem + ".sub")
        transcribe_opus_file(
            proc,
            model,
            device_str,
            ap,
            out_sub,
            batch_size=args.batch_size,
            chunk_length_s=args.chunk_length_s,
            chunk_overlap_s=args.chunk_overlap_s,
            max_sub_duration=args.max_sub_duration,
        )

if __name__ == "__main__":
    main()
