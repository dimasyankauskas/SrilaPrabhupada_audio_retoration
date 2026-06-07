# Compare — candidate ranking
Baseline: `720321SB_BOM_10min_v2.wav`  |  duration 600.0s

| candidate | hiss Δ (dB) | HF Δ (dB) | speech Δ (dB) | LRA Δ (LU) | LUFS out | RTF | score | hours/1k files | status |
|-----------|-------------|-----------|---------------|------------|----------|-----|-------|----------------|--------|
| c04_sox_classical | -1.0 | +13.6 | -8.8 | -1.5 | -16.1 | 0.02 | +6.73 | 4 h | ✅ |
| c10b_resemble_chunk_60 | +1.1 | -0.9 | -8.6 | -0.5 | -15.9 | 0.11 | -0.26 | 18 h | ✅ |
| c02_rnnoise_dedicated | +1.5 | -2.2 | -8.4 | -0.3 | -16.0 | 0.03 | -0.42 | 4 h | ✅ |
| c10d_resemble_overlap_2 | +1.1 | -1.2 | -8.6 | -0.5 | -15.9 | 0.11 | -0.58 | 19 h | ✅ |
| c07_resemble_denoise | +1.1 | -1.4 | -8.6 | -0.4 | -15.9 | 0.12 | -0.60 | 20 h | ✅ |
| c10a_resemble_baseline | +1.1 | -1.4 | -8.6 | -0.4 | -15.9 | 0.12 | -0.60 | 19 h | ✅ |
| c10e_resemble_preemph_85 | +1.1 | -1.4 | -8.6 | -0.4 | -15.9 | 0.11 | -0.60 | 18 h | ✅ |
| c10f_resemble_preemph_70 | +1.1 | -1.4 | -8.6 | -0.4 | -15.9 | 0.11 | -0.60 | 19 h | ✅ |
| c10c_resemble_chunk_15 | +1.0 | -1.3 | -8.6 | -0.5 | -15.9 | 0.12 | -0.69 | 20 h | ✅ |
| c16_studio_pipeline | -5.8 | +18.2 | -8.4 | -0.7 | -16.1 | 4.45 | -1.17 | 742 h | ✅ |
| c14_audiosr | -6.9 | +22.4 | -8.3 | -0.1 | -16.1 | 4.74 | -2.45 | 790 h | ✅ |
| c17_studio_voicefixer | -8.2 | +17.9 | -8.3 | +0.4 | -16.1 | 0.39 | -3.50 | 64 h | ✅ |
| c01_classical_ffmpeg | +4.3 | -13.0 | -8.4 | -0.3 | -16.0 | 0.07 | -4.85 | 11 h | ✅ |
| c15_mossformer2_48k | +0.4 | -11.0 | -8.6 | -0.2 | -15.9 | 0.10 | -10.64 | 16 h | ✅ |
| c12_deepfilternet | -1.1 | -11.9 | -8.2 | -0.4 | -16.3 | 0.04 | -14.70 | 7 h | ✅ |
| c05_hybrid_classical_ml | +0.6 | -44.9 | -8.3 | -0.2 | -16.1 | 0.25 | -44.12 | 41 h | ✅ |
| c11_facebook_denoiser_loudnorm | +0.2 | -44.3 | -8.3 | -0.4 | -16.1 | 0.36 | -44.44 | 60 h | ✅ |
| c08_resemble_enhance | -27.9 | +52.1 | +8.6 | -2.2 | -18.5 | 3.92 | -58.99 | 654 h | ✅ |
| c03_facebook_denoiser | +0.4 | -59.0 | +1.4 | +0.0 | -25.7 | 0.14 | -62.10 | 23 h | ✅ |
| c09_audio_maister | — | — | — | — | — | — | — | — | — | ⏭ skipped |
| c13_voicefixer | — | — | — | — | — | — | — | — | — | ⏭ skipped |

## Reading the table

- **hiss Δ** — how much the hiss-band vs speech-band ratio dropped in the candidate. (hiss_db − speech_db) is the gain-invariant hiss floor. Positive = hiss reduced relative to speech.
- **HF Δ** — change in 12–18 kHz energy. Positive = bandwidth extension worked (studio-quality HF added). The single biggest gap between 1970s tape and modern recordings.
- **speech Δ** — how much energy was removed from 300–3.4 kHz. Positive = removed speech (bad).
- **LRA Δ** — change in loudness range. Negative = dynamics got crushed (bad).
- **LUFS out** — measured output loudness. The podcast target is -16. Anything more than 3 LU away from -16 is penalized in the score (see formula).
- **RTF** — runtime ÷ audio duration. <1.0 is faster than realtime.
- **score** — composite: `2·hiss_ratio_Δ + min(HF_Δ, +20) − max(0,+speech_Δ) − 0.5·max(0,−LRA_Δ) − 0.1·|centroid−5000|/1000 − max(0,RTF−2) − 0.3·max(0,|LUFS+16|−3)`. Higher is better.
- hiss_ratio_Δ = (hiss_db − speech_db) in source minus (hiss_db − speech_db) in candidate. Positive = hiss reduced relative to speech (the gain-invariant measure).

## Recommendation

**Best candidate so far: `c04_sox_classical`** with score +6.73.

