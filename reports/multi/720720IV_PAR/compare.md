# Compare — candidate ranking
Baseline: `720720IV_PAR_full.wav`  |  duration 293.7s

| candidate | hiss Δ (dB) | HF Δ (dB) | speech Δ (dB) | LRA Δ (LU) | LUFS out | RTF | score | hours/1k files | status |
|-----------|-------------|-----------|---------------|------------|----------|-----|-------|----------------|--------|
| c02_rnnoise_dedicated | +5.0 | +0.9 | -7.8 | -1.5 | -17.4 | 0.03 | +9.79 | 2 h | ✅ |
| c10c_resemble_chunk_15 | +1.9 | +2.4 | -8.3 | -1.6 | -17.5 | 0.16 | +5.15 | 13 h | ✅ |
| c04_sox_classical | -1.4 | +11.6 | -8.4 | -4.7 | -16.8 | 0.02 | +4.63 | 2 h | ✅ |
| c10d_resemble_overlap_2 | +2.0 | +1.3 | -8.3 | -1.7 | -17.5 | 0.15 | +4.09 | 12 h | ✅ |
| c07_resemble_denoise | +2.0 | +1.2 | -8.3 | -1.6 | -17.5 | 0.17 | +4.08 | 14 h | ✅ |
| c10a_resemble_baseline | +2.0 | +1.2 | -8.3 | -1.6 | -17.5 | 0.15 | +4.08 | 13 h | ✅ |
| c10e_resemble_preemph_85 | +2.0 | +1.2 | -8.3 | -1.6 | -17.5 | 0.17 | +4.08 | 14 h | ✅ |
| c10f_resemble_preemph_70 | +1.9 | +1.2 | -8.3 | -1.6 | -17.5 | 0.23 | +3.96 | 19 h | ✅ |
| c10b_resemble_chunk_60 | +2.2 | -0.1 | -8.3 | -1.7 | -17.5 | 0.15 | +3.08 | 12 h | ✅ |
| c01_classical_ffmpeg | +8.8 | -14.6 | -7.8 | -1.2 | -17.4 | 0.07 | +1.96 | 6 h | ✅ |
| c17_studio_voicefixer | -5.8 | +37.2 | -7.3 | +7.5 | -17.5 | 0.40 | +1.70 | 33 h | ✅ |
| c12_deepfilternet | +5.6 | -10.3 | -7.9 | -0.2 | -17.3 | 0.04 | +0.41 | 3 h | ✅ |
| c15_mossformer2_48k | +3.8 | -11.1 | -8.4 | -0.5 | -17.2 | 0.10 | -4.15 | 8 h | ✅ |
| c16_studio_pipeline | -8.5 | +41.7 | -8.2 | -2.0 | -16.6 | 4.55 | -8.24 | 371 h | ✅ |
| c11_facebook_denoiser_loudnorm | +3.0 | -20.9 | -8.1 | -0.4 | -17.3 | 0.14 | -15.61 | 11 h | ✅ |
| c05_hybrid_classical_ml | +3.2 | -22.6 | -8.2 | +0.0 | -17.3 | 0.14 | -16.64 | 11 h | ✅ |
| c14_audiosr | -12.2 | +43.2 | -7.9 | -0.5 | -16.8 | 6.41 | -16.96 | 523 h | ✅ |
| c08_resemble_enhance | -19.2 | +57.4 | +5.3 | -6.1 | -18.8 | 4.11 | -39.79 | 335 h | ✅ |
| c03_facebook_denoiser | +5.7 | -53.2 | +0.5 | +5.5 | -26.2 | 0.12 | -44.90 | 10 h | ✅ |
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

**Best candidate so far: `c02_rnnoise_dedicated`** with score +9.79.

