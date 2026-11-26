#!/usr/bin/env python3
"""
Generate embedded 16kHz 16-bit mono signed PCM Chinese voice alert for M5Stack CoreS3.
Uses macOS built-in `say` + `afconvert` so zero extra TTS downloads are required.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def generate_voice_pcm(
    text: str,
    output_pcm: Path,
    voice: str = "Tingting",
    rate: int = 190,
    sample_rate: int = 16000,
) -> None:
    output_pcm = output_pcm.resolve()
    output_pcm.parent.mkdir(parents=True, exist_ok=True)

    if not shutil.which("say") or not shutil.which("afconvert"):
        raise RuntimeError("macOS 'say' and 'afconvert' utilities are required to generate the voice PCM.")

    with tempfile.TemporaryDirectory() as tmpdir:
        aiff_path = Path(tmpdir) / "alert.aiff"
        wav_path = Path(tmpdir) / "alert.wav"

        subprocess.run(
            ["say", "-v", voice, "-r", str(rate), "-o", str(aiff_path), text],
            check=True,
        )
        # Convert to 16kHz, 16-bit signed little-endian mono WAV, then strip 44-byte WAV header or emit raw PCM
        subprocess.run(
            [
                "afconvert",
                "-f", "WAVE",
                "-d", f"LEI16@{sample_rate}",
                "-c", "1",
                str(aiff_path),
                str(wav_path),
            ],
            check=True,
        )
        raw = wav_path.read_bytes()
        # Find the 'data' chunk in WAV to extract exact raw PCM payload
        idx = raw.find(b"data")
        if idx < 0 or idx + 8 > len(raw):
            raise RuntimeError("Invalid WAV output from afconvert (missing data chunk).")
        data_size = int.from_bytes(raw[idx + 4 : idx + 8], "little")
        pcm = raw[idx + 8 : idx + 8 + data_size]

        # Trim leading/trailing near-silence to keep Flash footprint compact & punchy
        samples = [
            int.from_bytes(pcm[i : i + 2], "little", signed=True)
            for i in range(0, len(pcm) - (len(pcm) % 2), 2)
        ]
        threshold = 220
        first = 0
        while first < len(samples) and abs(samples[first]) < threshold:
            first += 1
        last = len(samples) - 1
        while last > first and abs(samples[last]) < threshold:
            last -= 1
        pad = int(0.05 * sample_rate)
        first = max(0, first - pad)
        last = min(len(samples), last + pad)
        trimmed = b"".join(int(s).to_bytes(2, "little", signed=True) for s in samples[first:last])

        output_pcm.write_bytes(trimmed)
        duration_sec = len(trimmed) / (sample_rate * 2)
        print(
            f"[generate_audio] Generated '{text}' -> {output_pcm} "
            f"({len(trimmed) / 1024:.1f} KB, {duration_sec:.2f}s @ {sample_rate}Hz 16-bit mono PCM)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chinese voice alert PCM for CoreS3.")
    parser.add_argument("--text", default="警告，未佩戴安全帽，请立即佩戴", help="Alert text to synthesize.")
    parser.add_argument("--voice", default="Tingting", help="macOS say voice name.")
    parser.add_argument("--rate", type=int, default=195, help="Speech rate (words/min).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "firmware" / "main" / "assets" / "alert_zh.pcm",
        help="Output raw PCM path.",
    )
    args = parser.parse_args()
    generate_voice_pcm(args.text, args.output, voice=args.voice, rate=args.rate)


if __name__ == "__main__":
    main()
