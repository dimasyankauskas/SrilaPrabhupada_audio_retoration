# Compare — candidate ranking
Baseline: `760706AD_WASHINGTON_DC_10min.wav`  |  duration 600.0s

| candidate | hiss Δ (dB) | speech Δ (dB) | LRA Δ (LU) | LUFS out | RTF | score | hours/1k files | status |
|-----------|-------------|---------------|------------|----------|-----|-------|----------------|--------|
| c03_facebook_denoiser | +29.4 | +2.9 | +1.9 | -35.1 | 0.20 | +51.00 | 34 h | ✅ |
| c01_classical_ffmpeg | +21.1 | -14.5 | -0.7 | -17.5 | 0.08 | +41.77 | 14 h | ✅ |
| c05_hybrid_classical_ml | +12.3 | -14.8 | -2.2 | -17.6 | 0.22 | +23.41 | 37 h | ✅ |
| c11_facebook_denoiser_loudnorm | +11.3 | -14.9 | -2.2 | -17.7 | 0.18 | +21.43 | 30 h | ✅ |
| c02_rnnoise_dedicated | +7.9 | -14.6 | -0.6 | -17.6 | 0.03 | +15.51 | 4 h | ✅ |
| c10d_resemble_overlap_2 | +6.0 | -14.9 | -1.7 | -17.2 | 0.16 | +11.14 | 26 h | ✅ |
| c07_resemble_denoise | +5.9 | -14.9 | -1.6 | -17.2 | 0.15 | +11.06 | 25 h | ✅ |
| c10a_resemble_baseline | +5.9 | -14.9 | -1.6 | -17.2 | 0.26 | +11.06 | 44 h | ✅ |
| c10e_resemble_preemph_85 | +5.9 | -14.9 | -1.6 | -17.2 | 0.14 | +11.06 | 24 h | ✅ |
| c10f_resemble_preemph_70 | +5.9 | -14.9 | -1.6 | -17.2 | 0.13 | +11.06 | 21 h | ✅ |
| c10c_resemble_chunk_15 | +5.8 | -14.9 | -1.9 | -17.2 | 0.17 | +10.71 | 28 h | ✅ |
| c10b_resemble_chunk_60 | +5.1 | -14.9 | -1.8 | -17.2 | 0.22 | +9.38 | 37 h | ✅ |
| c08_resemble_enhance | -1.1 | +2.9 | -3.0 | -20.7 | 2.96 | -8.00 | 493 h | ✅ |
| c09_audio_maister | -4.0 | -12.7 | -0.5 | -18.8 | 0.99 | -8.15 | 166 h | ✅ |
| c04_sox_classical | -16.5 | -14.5 | -5.1 | -16.8 | 0.02 | -35.63 | 4 h | ✅ |

## Reading the table

- **hiss Δ** — how much energy was removed from the 5–12 kHz band. Positive = good.
- **speech Δ** — how much energy was removed from 300–3.4 kHz. Positive = removed speech (bad).
- **LRA Δ** — change in loudness range. Negative = dynamics got crushed (bad).
- **LUFS out** — measured output loudness. The podcast target is -16. Anything more than 3 LU away from -16 is penalized in the score (see formula).
- **RTF** — runtime ÷ audio duration. <1.0 is faster than realtime.
- **score** — composite: 2·hiss − max(0,+speech) − 0.5·max(0,−LRA) − max(0,RTF−2) − 0.3·max(0,|LUFS+16|−3). Higher is better.

## Recommendation

**Best candidate so far: `c03_facebook_denoiser`** with score +51.00.

