import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import re
import subprocess

import gdown
import torch
import torchaudio
from tqdm import tqdm

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoConfig,
    BitsAndBytesConfig,
)


LANGUAGE_CODES = {
    "Acehnese (Arabic script)": "ace_Arab",
    "Acehnese (Latin script)": "ace_Latn",
    "Afrikaans": "afr_Latn",
    "Akan": "aka_Latn",
    "Albanian (Tosk)": "als_Latn",
    "Amharic": "amh_Ethi",
    "Arabic (Mesopotamian)": "acm_Arab",
    "Arabic (Najdi)": "ars_Arab",
    "Arabic (North Levantine)": "apc_Arab",
    "Arabic (South Levantine)": "ajp_Arab",
    "Arabic (Ta’izzi-Adeni)": "acq_Arab",
    "Arabic (Tunisian)": "aeb_Arab",
    "Arabic (Modern Standard)": "arb_Arab",
    "Arabic (Modern Standard, Romanized)": "arb_Latn",
    "Armenian": "hye_Armn",
    "Assamese": "asm_Beng",
    "Asturian": "ast_Latn",
    "Awadhi": "awa_Deva",
    "Aymara (Central)": "ayr_Latn",
    "Bambara": "bam_Latn",
    "Balinese": "ban_Latn",
    "Banjar (Arabic script)": "bjn_Arab",
    "Banjar (Latin script)": "bjn_Latn",
    "Basque": "eus_Latn",
    "Bashkir": "bak_Cyrl",
    "Belarusian": "bel_Cyrl",
    "Bemba": "bem_Latn",
    "Bengali": "ben_Beng",
    "Bhojpuri": "bho_Deva",
    "Bosnian": "bos_Latn",
    "Burmese": "mya_Mymr",
    "Buginese": "bug_Latn",
    "Bulgarian": "bul_Cyrl",
    "Catalan": "cat_Latn",
    "Cebuano": "ceb_Latn",
    "Central Atlas Tamazight": "tzm_Tfng",
    "Central Kanuri (Arabic script)": "knc_Arab",
    "Central Kanuri (Latin script)": "knc_Latn",
    "Central Kurdish": "ckb_Arab",
    "Chhattisgarhi": "hne_Deva",
    "Chokwe": "cjk_Latn",
    "Chinese (Simplified)": "zho_Hans",
    "Chinese (Traditional)": "zho_Hant",
    "Croatian": "hrv_Latn",
    "Czech": "ces_Latn",
    "Danish": "dan_Latn",
    "Dari": "prs_Arab",
    "Dutch": "nld_Latn",
    "Dyula": "dyu_Latn",
    "Eastern Panjabi": "pan_Guru",
    "Eastern Yiddish": "ydd_Hebr",
    "Egyptian Arabic": "arz_Arab",
    "English": "eng_Latn",
    "Esperanto": "epo_Latn",
    "Estonian": "est_Latn",
    "Ewe": "ewe_Latn",
    "Faroese": "fao_Latn",
    "Fijian": "fij_Latn",
    "Finnish": "fin_Latn",
    "Fon": "fon_Latn",
    "French": "fra_Latn",
    "Friulian": "fur_Latn",
    "Galician": "glg_Latn",
    "Ganda": "lug_Latn",
    "Georgian": "kat_Geor",
    "German": "deu_Latn",
    "Greek": "ell_Grek",
    "Guarani": "grn_Latn",
    "Gujarati": "guj_Gujr",
    "Haitian Creole": "hat_Latn",
    "Halh Mongolian": "khk_Cyrl",
    "Hausa": "hau_Latn",
    "Hebrew": "heb_Hebr",
    "Hindi": "hin_Deva",
    "Icelandic": "isl_Latn",
    "Igbo": "ibo_Latn",
    "Ilocano": "ilo_Latn",
    "Indonesian": "ind_Latn",
    "Italian": "ita_Latn",
    "Japanese": "jpn_Jpan",
    "Javanese": "jav_Latn",
    "Jingpho": "kac_Latn",
    "Kabiyè": "kbp_Latn",
    "Kabyle": "kab_Latn",
    "Kamba": "kam_Latn",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic script)": "kas_Arab",
    "Kashmiri (Devanagari script)": "kas_Deva",
    "Kazakh": "kaz_Cyrl",
    "Khmer": "khm_Khmr",
    "Kikongo": "kon_Latn",
    "Kikuyu": "kik_Latn",
    "Kimbundu": "kmb_Latn",
    "Kinyarwanda": "kin_Latn",
    "Korean": "kor_Hang",
    "Kyrgyz": "kir_Cyrl",
    "Lao": "lao_Laoo",
    "Latgalian": "ltg_Latn",
    "Ligurian": "lij_Latn",
    "Limburgish": "lim_Latn",
    "Lingala": "lin_Latn",
    "Lithuanian": "lit_Latn",
    "Lombard": "lmo_Latn",
    "Luba-Kasai": "lua_Latn",
    "Luxembourgish": "ltz_Latn",
    "Luo": "luo_Latn",
    "Maithili": "mai_Deva",
    "Magahi": "mag_Deva",
    "Malayalam": "mal_Mlym",
    "Maltese": "mlt_Latn",
    "Macedonian": "mkd_Cyrl",
    "Meitei (Bengali script)": "mni_Beng",
    "Minangkabau (Arabic script)": "min_Arab",
    "Minangkabau (Latin script)": "min_Latn",
    "Moroccan Arabic": "ary_Arab",
    "Mossi": "mos_Latn",
    "Nepali": "npi_Deva",
    "Northern Kurdish": "kmr_Latn",
    "Northern Sotho": "nso_Latn",
    "Northern Uzbek": "uzn_Latn",
    "Norwegian Bokmål": "nob_Latn",
    "Norwegian Nynorsk": "nno_Latn",
    "Nuer": "nus_Latn",
    "Nyanja": "nya_Latn",
    "Occitan": "oci_Latn",
    "Odia": "ory_Orya",
    "Pangasinan": "pag_Latn",
    "Papiamento": "pap_Latn",
    "Pashto (Southern)": "pbt_Arab",
    "Persian (Western)": "pes_Arab",
    "Polish": "pol_Latn",
    "Portuguese": "por_Latn",
    "Quechua (Ayacucho)": "quy_Latn",
    "Romanian": "ron_Latn",
    "Rundi": "run_Latn",
    "Russian": "rus_Cyrl",
    "Sango": "sag_Latn",
    "Sardinian": "srd_Latn",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Scottish Gaelic": "gla_Latn",
    "Sicilian": "scn_Latn",
    "Shan": "shn_Mymr",
    "Sinhala": "sin_Sinh",
    "Slovak": "slk_Latn",
    "Slovenian": "slv_Latn",
    "Somali": "som_Latn",
    "Spanish": "spa_Latn",
    "Sotho (Southern)": "sot_Latn",
    "Swati": "ssw_Latn",
    "Sundanese": "sun_Latn",
    "Swahili": "swh_Latn",
    "Swedish": "swe_Latn",
    "Silesian": "szl_Latn",
    "Tajik": "tgk_Cyrl",
    "Tamasheq (Latin script)": "taq_Latn",
    "Tamasheq (Tifinagh script)": "taq_Tfng",
    "Tamil": "tam_Taml",
    "Tatar": "tat_Cyrl",
    "Telugu": "tel_Telu",
    "Thai": "tha_Thai",
    "Tigrinya": "tir_Ethi",
    "Tok Pisin": "tpi_Latn",
    "Tsonga": "tso_Latn",
    "Tswana": "tsn_Latn",
    "Turkish": "tur_Latn",
    "Turkmen": "tuk_Latn",
    "Tumbuka": "tum_Latn",
    "Twi": "twi_Latn",
    "Ukrainian": "ukr_Cyrl",
    "Umbundu": "umb_Latn",
    "Urdu": "urd_Arab",
    "Uyghur": "uig_Arab",
    "Venetian": "vec_Latn",
    "Vietnamese": "vie_Latn",
    "Waray": "war_Latn",
    "Welsh": "cym_Latn",
    "Wolof": "wol_Latn",
    "Xhosa": "xho_Latn",
    "Yoruba": "yor_Latn",
    "Yue Chinese": "yue_Hant",
    "Zulu": "zul_Latn",
}


def extract_gdrive_file_id(file_link: str) -> str:
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", file_link)
    if not match:
        raise ValueError("Could not extract Google Drive File ID from the provided link.")
    return match.group(1)


def download_gdrive_file(file_id: str, output: Optional[str] = None) -> str:
    filename = gdown.download(id=file_id, output=output, quiet=False)
    if not filename:
        raise RuntimeError("Download failed. Please check the link.")
    return filename


def download_file_if_possible(path: Path):
    try:
        colab_files = __import__("google.colab", fromlist=["files"]).files
        colab_files.download(str(path))
    except Exception:
        print(f"SRT ready at: {path} (auto-download only works in Google Colab)")


def ffprobe_duration(path: Path) -> float:
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    return float(process.stdout.decode("utf-8").strip())


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
    with open(out_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines))


def build_whisper(model_name: str = "openai/whisper-large-v3", device: int = 0):
    proc = WhisperProcessor.from_pretrained(model_name)
    # Define the 8-bit config here
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True  # The argument goes HERE
    )
    model = WhisperForConditionalGeneration.from_pretrained(
      model_name, 
      use_safetensors=False, 
      device_map="auto",
      quantization_config=bnb_config,
      )
    
    device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    model.to(device_str)
    model.eval()
    return proc, model, device_str


class Translator:
    def __init__(self):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        model_name = "facebook/nllb-200-distilled-1.3B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        config = AutoConfig.from_pretrained(model_name)
        config.tie_word_embeddings = False
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config, use_safetensors=False, torch_dtype=torch.float16)
        self.model.to(self.device)

    def get_device(self):
        return self.device.type

    def __call__(self, text: list, src_code: str, tgt_code: str) -> list:
        self.tokenizer.src_lang = src_code
        self.tokenizer.tgt_lang = tgt_code

        input_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # print(input_tokens.input_ids.shape)
        # input_tokens = input_tokens.input_ids[0].cpu().numpy().tolist()

        outputs = self.model.generate(
            input_ids=torch.tensor(input_tokens.input_ids), #.to(self.device),
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_code),
            max_length=len(input_tokens) + 50,
            num_return_sequences=1,
            num_beams=5,
            no_repeat_ngram_size=4,
            renormalize_logits=True,
        )

        decoded = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        return decoded


def transcribe_audio_file(
    proc: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device_str: str,
    input_path: Path,
    out_srt: Path,
    batch_size: int = 8,
    chunk_length_s: float = 30,
    chunk_overlap_s: float = 0.5,
    max_sub_duration: float = 5.0,
    translator: Optional[Translator] = None,
    src_code: Optional[str] = None,
    tgt_code: Optional[str] = None,
    debug: bool = False,
):
    work_dir = Path(".tmp")
    work_dir.mkdir(parents=True, exist_ok=True)

    wav_padded = work_dir / (input_path.stem + "_pad.wav")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-filter_complex",
        "[0:a]atrim=0:30,asetpts=N/SR/TB[a0];[a0][0:a]concat=n=2:v=0:a=1[a]",
        "-map",
        "[a]",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_padded),
    ]

    if debug:
        ffmpeg_cmd.insert(4, "-t")
        ffmpeg_cmd.insert(5, "180")  # Limit to first 3 minutes if debug is enabled

    subprocess.run(ffmpeg_cmd, check=True)

    total_dur = ffprobe_duration(wav_padded)
    ranges = chunk_ranges(total_dur, chunk_length_s, chunk_overlap_s)

    chunk_files = []
    for idx, (start, length) in enumerate(ranges):
        chunk_wav = work_dir / f"{input_path.stem}_chunk{idx:04d}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-t",
                str(length),
                "-i",
                str(wav_padded),
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(chunk_wav),
            ],
            check=True,
        )
        chunk_files.append((chunk_wav, start, length))

    global_segments = []
    dtype = torch.float16 if "cuda" in device_str else torch.float32

    for i in tqdm(range(0, len(chunk_files), batch_size), desc=f"Chunks {input_path.stem}", leave=False):
        batch = chunk_files[i : i + batch_size]

        inputs_list = []
        for chunk_path, chunk_start, chunk_len in batch:
            waveform, sample_rate = torchaudio.load(str(chunk_path))
            if waveform.ndim == 2:
                waveform = waveform.mean(dim=0)
            audio_np = waveform.numpy()
            proc_inputs = proc(audio_np, sampling_rate=sample_rate, return_tensors="pt")
            inputs_list.append((proc_inputs.input_features, chunk_start, chunk_len))

        batch_features = torch.cat([input_features for input_features, _, _ in inputs_list], dim=0)
        batch_features = batch_features.to(device_str).to(dtype)

        with torch.inference_mode():
            gen_out = model.generate(
                batch_features,
                max_length=448,
                return_dict_in_generate=True,
                return_timestamps=True,
            )

        segments_obj = gen_out.get("segments", None) if isinstance(gen_out, dict) else getattr(gen_out, "segments", None)

        if segments_obj is None:
            texts = []
            if isinstance(gen_out, dict) and "text" in gen_out:
                texts_val = gen_out["text"]
                if isinstance(texts_val, str):
                    texts = [texts_val]
                else:
                    texts = list(texts_val)
            elif hasattr(gen_out, "sequences"):
                texts = proc.tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)

            if len(texts) < len(inputs_list):
                texts.extend([""] * (len(inputs_list) - len(texts)))

            for idx_item, (_, chunk_start, chunk_len) in enumerate(inputs_list):
                text = texts[idx_item] if idx_item < len(texts) else ""
                global_segments.append({"start": chunk_start, "end": chunk_start + chunk_len, "text": text})
            continue

        for idx_item, segmented_output in enumerate(segments_obj):
            _, chunk_start, chunk_len = inputs_list[idx_item]

            if not segmented_output:
                global_segments.append({"start": chunk_start, "end": chunk_start + chunk_len, "text": ""})
                continue

            seg_texts = []
            seg_starts = []
            for seg in segmented_output:
                tokens = seg.get("tokens", None)
                toks = tokens.cpu().tolist() if isinstance(tokens, torch.Tensor) else tokens
                text = proc.tokenizer.decode(toks, skip_special_tokens=True).strip() if toks is not None else seg.get("text", "").strip()
                seg_texts.append(text)
                start_val = seg.get("start", None)
                if isinstance(start_val, torch.Tensor):
                    start_val = float(start_val.cpu().item())
                seg_starts.append(float(start_val) if start_val is not None else 0.0)

            last_seg = segmented_output[-1]
            last_end = last_seg.get("end", None)
            if isinstance(last_end, torch.Tensor):
                last_end = float(last_end.cpu().item())
            if last_end is None:
                last_end = chunk_len

            for idx_seg in range(len(seg_texts)):
                seg_start = seg_starts[idx_seg]
                seg_end = seg_starts[idx_seg + 1] if idx_seg + 1 < len(seg_texts) else last_end
                global_segments.append({"start": seg_start + chunk_start, "end": seg_end + chunk_start, "text": seg_texts[idx_seg]})

    adjusted_segments = []
    for seg in global_segments:
        if seg["start"] < 30.0:
            continue
        new_start = seg["start"] - 30.0
        new_end = seg["end"] - 30.0
        if new_end < new_start:
            new_end = new_start
        if max_sub_duration is not None and max_sub_duration > 0 and new_end - new_start > max_sub_duration:
            new_end = new_start + max_sub_duration

        adjusted_segments.append({"start": new_start, "end": new_end, "text": seg["text"]})

    if translator is not None and src_code and tgt_code:
        translatable_indices = [idx for idx, seg in enumerate(adjusted_segments) if seg["text"].strip()]
        for i in tqdm(range(0, len(translatable_indices), batch_size)):
            batch_indices = translatable_indices[i : i + batch_size]
            batch_texts = [adjusted_segments[idx]["text"] for idx in batch_indices]
            translated_batch = translator(batch_texts, src_code, tgt_code)
            for idx, translated_text in zip(batch_indices, translated_batch):
                adjusted_segments[idx]["text"] = translated_text

    write_srt(adjusted_segments, out_srt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ASR subtitle generation with optional translation.")
    parser.add_argument("--input_path", default=None, help="Input media path (audio or video). If provided, only this file is processed.")
    parser.add_argument("--input_dir", default="input", help="Input directory. When --input_path is not provided, all files in this directory are processed.")
    parser.add_argument("--output", default=None, help="Output .srt path for single-file mode.")
    parser.add_argument("--output_dir", default="output", help="Output directory for directory mode.")
    parser.add_argument("--source_lang", default=None, help="Source language name, e.g. English")
    parser.add_argument("--target_lang", default=None, help="Target language name, e.g. Turkish")
    parser.add_argument("--model", default="openai/whisper-large-v3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--chunk_length_s", type=float, default=30)
    parser.add_argument("--chunk_overlap_s", type=float, default=0.5)
    parser.add_argument("--max_sub_duration", type=float, default=5.0)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    if args.debug:
        # args.model = "openai/whisper-tiny"
        print("Debug mode enabled: using smaller model and processing only first 3 minutes of audio.")

    input_paths = []
    if args.input_path:
        single_input_path = Path(args.input_path)
        if not single_input_path.exists() or not single_input_path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {single_input_path}")
        input_paths = [single_input_path]
    else:
        input_dir_path = Path(args.input_dir)
        if not input_dir_path.exists() or not input_dir_path.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir_path}")
        input_paths = sorted([path for path in input_dir_path.iterdir() if path.is_file()])

    if not input_paths:
        raise ValueError("No input files found to process.")

    source_lang_name = args.source_lang.capitalize() if args.source_lang else None
    target_lang_name = args.target_lang.capitalize() if args.target_lang else None

    if source_lang_name and source_lang_name not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported source language name: {source_lang_name}")
    if target_lang_name and target_lang_name not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported target language name: {target_lang_name}")

    source_lang = LANGUAGE_CODES.get(source_lang_name) if source_lang_name else None
    target_lang = LANGUAGE_CODES.get(target_lang_name) if target_lang_name else None

    translator = None
    if bool(source_lang) ^ bool(target_lang):
        raise ValueError("Provide both --source_lang and --target_lang together, or leave both as None, with valid language names.")
    if source_lang and target_lang:
        if source_lang == target_lang:
            print("Translation disabled because source and target languages are the same.")
            source_lang = None
            target_lang = None
        else:
            print(f"Translation enabled: {source_lang_name} -> {target_lang_name}")
            translator = Translator()
            print(f"Translator loaded on: {translator.get_device()}")
    else:
        print("Translation disabled.")

    if args.input_path and args.output:
        output_paths = [Path(args.output)]
    elif args.input_path:
        output_paths = [Path(args.input_path).with_suffix(".srt")]
    else:
        output_dir_path = Path(args.output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_paths = [output_dir_path / f"{input_path.stem}.srt" for input_path in input_paths]

    print("Building Whisper model...")
    proc, model, device_str = build_whisper(model_name=args.model, device=args.device)
    print(f"Whisper ready on: {device_str}")

    for input_path, output_path in tqdm(list(zip(input_paths, output_paths)), desc="Files"):
        print(f"Processing: {input_path}")
        transcribe_audio_file(
            proc=proc,
            model=model,
            device_str=device_str,
            input_path=input_path,
            out_srt=output_path,
            batch_size=args.batch_size,
            chunk_length_s=args.chunk_length_s,
            chunk_overlap_s=args.chunk_overlap_s,
            max_sub_duration=args.max_sub_duration,
            translator=translator,
            src_code=source_lang,
            tgt_code=target_lang,
            debug=args.debug,
        )
        print(f"SRT written to: {output_path}")

