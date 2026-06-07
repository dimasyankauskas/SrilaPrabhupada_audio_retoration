# Compare — candidate ranking
Baseline: `680323MW_SAN_FRANCISCO_10min.wav`  |  duration 600.0s

| candidate | hiss Δ (dB) | HF Δ (dB) | speech Δ (dB) | LRA Δ (LU) | LUFS out | RTF | score | hours/1k files | status |
|-----------|-------------|-----------|---------------|------------|----------|-----|-------|----------------|--------|
| c01_classical_ffmpeg | +5.7 | +26.5 | -7.4 | -3.4 | -18.2 | 0.07 | +29.28 | 11 h | ✅ |
| c10b_resemble_chunk_60 | +3.3 | +35.2 | -7.4 | -1.3 | -17.8 | 0.12 | +25.61 | 20 h | ✅ |
| c10c_resemble_chunk_15 | +3.2 | +35.6 | -7.5 | -1.3 | -17.9 | 0.12 | +25.32 | 20 h | ✅ |
| c07_resemble_denoise | +3.4 | +35.4 | -7.5 | -3.2 | -17.8 | 0.12 | +24.76 | 20 h | ✅ |
| c10a_resemble_baseline | +3.4 | +35.4 | -7.5 | -3.2 | -17.8 | 0.12 | +24.76 | 20 h | ✅ |
| c10e_resemble_preemph_85 | +3.4 | +35.4 | -7.5 | -3.2 | -17.8 | 0.11 | +24.76 | 19 h | ✅ |
| c10f_resemble_preemph_70 | +3.4 | +35.4 | -7.5 | -3.2 | -17.8 | 0.12 | +24.76 | 19 h | ✅ |
| c10d_resemble_overlap_2 | +3.2 | +35.5 | -7.5 | -2.8 | -17.8 | 0.12 | +24.64 | 19 h | ✅ |
| c12_deepfilternet | +4.2 | +18.6 | -7.0 | -6.9 | -17.0 | 0.05 | +23.28 | 8 h | ✅ |
| c15_mossformer2_48k | +2.5 | +25.9 | -6.9 | -4.2 | -17.1 | 0.30 | +22.40 | 51 h | ✅ |
| c02_rnnoise_dedicated | +1.7 | +26.8 | -7.4 | -4.6 | -18.4 | 0.03 | +20.72 | 4 h | ✅ |
| c11_facebook_denoiser_loudnorm | +2.4 | +18.4 | -7.0 | -4.6 | -16.9 | 0.31 | +20.42 | 51 h | ✅ |
| c05_hybrid_classical_ml | +1.8 | +18.1 | -6.9 | -4.9 | -16.9 | 0.21 | +18.94 | 36 h | ✅ |
| c04_sox_classical | -3.9 | +31.9 | -9.4 | -9.4 | -17.6 | 0.02 | +7.19 | 4 h | ✅ |
| c03_facebook_denoiser | +2.8 | -7.0 | +0.5 | -1.9 | -24.7 | 0.14 | -4.99 | 23 h | ✅ |
| c17_studio_voicefixer | -22.9 | +81.0 | -7.3 | -4.4 | -17.3 | 0.91 | -34.89 | 152 h | ✅ |
| c16_studio_pipeline | -32.0 | +88.7 | -7.2 | -6.2 | -16.8 | 5.61 | -58.88 | 935 h | ✅ |
| c14_audiosr | -38.6 | +95.2 | -5.7 | -7.8 | -18.2 | 8.07 | -76.91 | 1346 h | ✅ |
| c08_resemble_enhance | -40.5 | +104.7 | +6.4 | -10.5 | -19.3 | 3.41 | -85.66 | 568 h | ✅ |
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

**Best candidate so far: `c01_classical_ffmpeg`** with score +29.28.

