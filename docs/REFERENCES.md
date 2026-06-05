# References

Sources used to build this evaluation harness. Live links, dates noted where
the URL includes one. All read in June 2026.

## Open-source ML denoisers (tried in the harness)

- **facebookresearch/denoiser** — Defossez et al., "Real Time Speech Enhancement in the Waveform Domain", Interspeech 2020. 135 MB PyTorch model, MIT, real-time on laptop CPU. [github.com/facebookresearch/denoiser](https://github.com/facebookresearch/denoiser)
- **Demucs (v4, HTDemucs)** — Rouard, Massa, Défossez, "Hybrid Transformers for Music Source Separation", ICASSP 2023. MIT. Audio-to-audio source separation. The M1-friendly port is [github.com/iky1e/demucs-mlx-swift](https://github.com/iky1e/demucs-mlx-swift) and HF: [mlx-community/demucs-mlx](https://huggingface.co/mlx-community/demucs-mlx).
- **Resemble Enhance** — Resemble AI, MIT, denoiser + enhancer trained on 44.1 kHz speech. [github.com/resemble-ai/resemble-enhance](https://github.com/resemble-ai/resemble-enhance), HF: [ResembleAI/resemble-enhance](https://huggingface.co/ResembleAI/resemble-enhance).
- **RNNoise** — Valin, "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement", 2018. C library, BSD-3. [jmvalin.ca/demo/rnnoise](https://jmvalin.ca/demo/rnnoise). Ffmpeg wraps it as `arnndn`.

## ffmpeg / sox references

- ffmpeg `arnndn` — uses RNNoise model. ffmpeg 8.0.1 confirmed on this machine.
- ffmpeg `anlmdn` — non-local means denoise. Documented at [ffmpeg.org/ffmpeg-filters.html#anlmdn](https://ffmpeg.org/ffmpeg-filters.html#anlmdn-1).
- ffmpeg `adeclick`, `deesser` — impulse and sibilance repair.
- ffmpeg `loudnorm` — EBU R128 broadcast loudness normalizer.
- sox `noisered`, `sinc`, `fir`, `compand` — see `man sox`.

## Tape characteristics (1970s cassette)

- **NAB / IEC1 / IEC2 equalization curves** — Richard Hess's notes at [richardhess.com/notes/formats/magnetic-media/magnetic-tapes/analog-audio/equalization/](https://richardhess.com/notes/formats/magnetic-media/magnetic-tapes/analog-audio/equalization/). NAB (North America) and IEC1/IEC2 (Europe) differ; the wrong one over-boosts bass.
- **Tape EQ comparison tool** — [eq.polartape.com](https://eq.polartape.com/) lets you compute correction values for IEC1/NAB/cassette at different tape speeds.
- **Prague 1981 standard** — [tapeheads.net discussion](https://www.tapeheads.net/threads/iec-equalization-pre-and-post-1981-prague-why.112416) on why cassette decks changed their replay EQ curve mid-1981 (Nakamichi-led push for higher treble headroom).
- **Dolby B** — broadband compander, dominant in 70s consumer cassettes. If the original was recorded with Dolby B and digitized without decoding, the highs are harsh and noisy. We treat the source as unknown and apply mild de-essing instead of a full Dolby decode.

## Apple Silicon M1 specifics

- **PyTorch MPS backend** — [developer.apple.com/metal/pytorch](https://developer.apple.com/metal/pytorch). macOS 14+, Python 3.10+, `pip install torch torchvision torchaudio`. `torch.backends.mps.is_available()` reports availability.
- **MLX** — Apple's own NumPy-like framework for Apple Silicon, 50–70% faster than MPS for inference. [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx).
- **M1 / 16 GB unified memory** — 68 GB/s bandwidth, 7–8 GPU cores. Audio ML models under ~500 MB fit comfortably. The denoiser model (135 MB) leaves headroom for the audio itself.
- Demucs on M1 via MLX: [medium article](https://medium.com/@andradeolivier/i-ported-demucs-to-apple-silicon-it-separates-a-7-minute-song-in-12-seconds-6c4e5cffb5c3) — 34× realtime on M4 Max. M1 will be ~3–5× realtime.

## Source file context

- **760706AD-WASHINGTON DC.MP3** — Srila Prabhupada lecture, Washington DC temple, 6 July 1976. The `AD` = A.C. Bhaktivedanta Swami. Cross-referenced against the [Vedabase transcripts list](https://vedabase.io/en/library/transcripts/?type=Bhagavad-gita&type=Conversation&location=Washington%2C+D.C.) which shows an "Evening Darśana July 6, 1976 — Washington, D.C." and a "Room Conversation With Scientists July 6, 1976 — Washington, D.C." on that date.

## Score / quality metrics

- **EBU R128** — broadcast loudness standard, integrated (LUFS), true-peak (dBTP), LRA (loudness range). Used via ffmpeg `loudnorm`.
- **Band energy** — spectral energy in user-defined bands (5–12 kHz for hiss, 300 Hz–3.4 kHz for speech, 20–200 Hz for rumble). No clean reference required.
- **Spectral centroid** — frequency-weighted mean of magnitude spectrum. Falls when hiss is suppressed.
- **Zero-crossing rate** — fraction of consecutive samples that cross zero. High values correlate with clicky/noisy audio.
- **DNSMOS P.835** — Microsoft DNS Challenge no-reference MOS predictor. Future enhancement; would give a single perceptual-quality score per output. Repo: [github.com/microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge).

## Why compare-first (not pick-first)

We do not commit to a pipeline up front. Every candidate runs on the
same source, gets the same metrics, and the ranking is data-driven. The
current ranking in `reports/compare.md` is the source of truth for
"which approach is best on this sample." The AGENTS.md states this
explicitly; the agent should never declare a winner before the table
has been generated.
