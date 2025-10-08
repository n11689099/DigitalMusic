# DigitalMusicAnalysis Parallelisation Report
**Student:** {STUDENT_NAME}

**Unit Code:** {UNIT_CODE}

## Abstract
DigitalMusicAnalysis is a WPF/.NET Core 3.1 application that aligns a recorded performance with a MusicXML score by computing a spectrogram, detecting note onsets, estimating pitch content, and synchronising events before rendering feedback. The original implementation executed these stages sequentially, dominated by the short-time Fourier transform (STFT) pipeline and frame-wise feature extraction. I refactored the compute-bound kernels to exploit thread-level data parallelism and SIMD vectorisation while preserving the existing UI workflow. Measured on a {CPU_MODEL} CPU, the optimised build achieves a headline end-to-end speedup of **{TOTAL_SPEEDUP}×** for the 45 s reference workload, with STFT throughput improving most dramatically. The report explains the baseline, parallel design choices, performance evaluation, and correctness validation. It concludes that frame-level parallelism combined with vectorised inner loops delivers scalable gains on modern multicore hardware and provides actionable next steps for further acceleration.

## Application Overview (Sequential Baseline)
DigitalMusicAnalysis loads a WAV recording and its corresponding MusicXML score, computes a time–frequency representation via STFT, derives onset and pitch histograms, aligns detected musical events with score positions, and finally renders visual feedback and playback controls in the WPF UI. The sequential pipeline consists of:

1. **Load & Preprocess:** Read the WAV samples with NAudio, normalise amplitude, parse MusicXML, and prepare analysis buffers.
2. **STFT Spectrogram:** Apply a window (e.g., Hann, size {FFT_SIZE}, hop {HOP_SIZE}) across the audio to generate complex FFT frames.
3. **Onset & Pitch Extraction:** Compute magnitudes, spectral flux, and histogram features per frame to detect onsets and estimate pitch class energy.
4. **Score Alignment:** Align detected events to score note timestamps via dynamic time warping-like heuristics.
5. **Render & Playback:** Update WPF visual layers and optionally play synchronised audio.

The workload is compute-intensive because STFT requires thousands of FFTs across 30–60 s of audio (≈ {NUM_FRAMES} frames at 44.1 kHz, hop {HOP_SIZE}); each frame performs windowing, FFT, magnitude, and histogram accumulation. Onset and pitch detection iterate over every frame and frequency bin, while alignment evaluates per-note comparisons. Together these stages account for >80% of runtime.

Baseline timing used the Release build, running the fixed reference audio/score three times outside the debugger. Each stage logged durations via `System.Diagnostics.Stopwatch` into CSV (`timestamp,commit,stage,threads,simd,duration_ms,notes,frames`). Median values defined the baseline.

**Table 1. Baseline per-stage timings (placeholder).**

| Stage | Duration (ms) | % of Total |
|-------|---------------|------------|
| ReadWav | {T_READWAV} | {P_READWAV} |
| STFT | {T_STFT} | {P_STFT} |
| OnsetDetect | {T_ONSET} | {P_ONSET} |
| PitchEstimate | {T_PITCH} | {P_PITCH} |
| AlignToScore | {T_ALIGN} | {P_ALIGN} |
| Render | {T_RENDER} | {P_RENDER} |
| Total | {T_TOTAL} | 100% |

## Hardware & Software Environment
- **CPU:** {CPU_MODEL}, {NUM_CORES} cores / {NUM_THREADS} threads, base clock {BASE_CLOCK} GHz (Turbo enabled; runs locked to High Performance power plan).
- **Memory:** {RAM_GB} GB DDR, dual-channel. Memory bandwidth affects FFT throughput when frames exceed cache capacity.
- **Operating System:** {OS_VERSION}.
- **Software Stack:** .NET Core {DOTNET_VER}, Visual Studio {VS_VER}, NAudio {NAUDIO_VER}. Command-line analysis mode invoked via `dotnet run --project DigitalMusicAnalysis`.
- **Profiling Tooling:** {PROFILER} for sampling CPU hotspots; `Stopwatch` instrumentation for per-stage timings; CSV logs analysed in Python/pandas.
- **Threading Model:** .NET ThreadPool with affinity left default; hyper-threading ({NUM_THREADS}/{NUM_CORES}) considered when analysing scaling.

## Profiling & Hotspot Analysis
Sampling profiles and instrumentation indicated that the STFT pipeline (`timefreq.ProcessFrame`) and subsequent magnitude/histogram loops dominate CPU time. FFT windowing and magnitude accumulation represent the tight inner loops. Onset detection (`timefreq.ComputeSpectralFlux`) and pitch histogram normalisation also consume significant time, while I/O and rendering are comparatively minor. These kernels iterate independently over frames or bins, making them ideal for data-parallel decomposition.

**Figure 1. CPU usage and top functions (placeholder).** This figure should illustrate the profiler’s top inclusive time list, highlighting STFT and histogram routines as the primary hotspots.

## Parallelisation Strategy
### 5.1 Data Parallelism (Threads)
- **Frame-Level STFT:** `Parallel.For` splits the frame range among worker threads. Each thread reuses thread-local FFT plans, window buffers, and magnitude arrays to avoid contention. A final reduction aggregates per-thread onset/pitch statistics.
- **Onset & Histogram Kernels:** Per-frame spectral flux and histogram computations run in the same `Parallel.For`. Thread-local bins accumulate energy; after the parallel loop, a reduction combines bins into global structures, eliminating false sharing.
- **Pipeline Coordination:** WAV decoding remains sequential, after which the analysis operates in batches. If streaming is required, a bounded blocking collection can feed frames into worker threads (producer–consumer), but for the fixed-size workload, bulk parallel loops suffice.
- **Thread Configuration:** Thread counts are user-selectable (`--threads N`). The application caps N at the number of logical cores to avoid oversubscription.

### 5.2 Vectorisation (SIMD)
- **Windowing & Magnitude:** Inner loops employ `System.Numerics.Vector<float>` (mapping to AVX/AVX2) to process multiple samples simultaneously. Window coefficients are aligned, and loop tails fall back to scalar code.
- **Dot Products & Reductions:** Spectral flux differences and histogram updates use SIMD fused multiply-add when available. For CPUs without AVX2, code gracefully degrades to SSE2-sized vectors.
- **Alignment & Safety:** Buffers align to 32 bytes via pinned arrays; tail handling ensures deterministic results. SIMD paths gated by the `--simd on|off` flag for ablation runs.

### 5.3 Correctness Guarantees
- **Determinism:** Parallel loops operate on disjoint frame ranges and thread-local accumulators, eliminating data races. Reductions use deterministic ordering (e.g., summing thread-local arrays in index order).
- **Numerical Stability:** Differences between scalar and SIMD code stem solely from reassociation within IEEE-754 tolerance. Validation thresholds (onset ±20 ms, pitch error ≤ ±1 semitone, alignment ±1 note) ensure outputs match musical expectations.
- **Randomness:** No stochastic components exist; repeated runs with the same inputs produce identical CSV logs.

## Experimental Methodology
- **Workload:** Reference audio is a mono 44.1 kHz WAV of duration {REF_DURATION} s paired with a MusicXML score of {REF_MEASURES} measures. Window length {FFT_SIZE}, hop {HOP_SIZE}, FFT size {FFT_SIZE} (power-of-two), Hann window.
- **Thread Counts:** Runs executed with threads N ∈ {1, 2, 4, 8, …, {MAX_THREADS}} (capped at physical cores). Each configuration repeated three times; the median duration reported. SIMD flag toggled for ablation.
- **Environment Control:** Release build executed via CLI with `--repeat 3`; machine idle; network and background services minimised; power profile fixed.
- **Metrics:** Collected per-stage duration (ms), total time, frames processed, detected notes. Speedup and efficiency computed as:
  - \(S(N) = T_1 / T_N\)
  - \(E(N) = S(N) / N\)
- **Data Handling:** CSV logs aggregated in pandas; plots generated with matplotlib scripts (see addendum). Commit hash recorded in every run for traceability.

## Results
### 7.1 End-to-End Speedup
**Figure 2. Total speedup vs. threads (placeholder).** Plot the median speedup \(S(N)\) for total analysis time.

**Table 2. End-to-end runtime, speedup, and efficiency (placeholder).**

| Threads (N) | Total Time T_N (ms) | Speedup S(N) | Efficiency E(N) |
|-------------|---------------------|--------------|------------------|
| 1 | {T1_TOTAL} | 1.00 | 1.00 |
| 2 | {T2_TOTAL} | {S2_TOTAL} | {E2_TOTAL} |
| 4 | {T4_TOTAL} | {S4_TOTAL} | {E4_TOTAL} |
| 8 | {T8_TOTAL} | {S8_TOTAL} | {E8_TOTAL} |
| {MAX_THREADS} | {TMAX_TOTAL} | {SMAX_TOTAL} | {EMAX_TOTAL} |

Headline observation: total speedup saturates near {SATURATION_THREADS} threads because serial components (I/O, alignment) and memory bandwidth limit scalability beyond that point.

### 7.2 Per-Stage Improvements
**Figure 3. Per-stage runtime comparison (placeholder).** Stacked bars contrasting 1-thread vs. {BEST_THREADS}-thread timings.

`STFT` and `OnsetDetect` show near-linear scaling thanks to balanced per-frame workloads and thread-local buffers. `AlignToScore` and `Render` remain mostly serial, constraining overall speedup.

### 7.3 SIMD Ablation
**Table 3. SIMD impact on hottest kernel (placeholder).**

| Configuration | Kernel Time (ms) | Δ vs Scalar | End-to-End Time (ms) | Δ vs Scalar |
|---------------|------------------|-------------|----------------------|-------------|
| SIMD Off | {KERNEL_SCALAR} | – | {TOTAL_SCALAR} | – |
| SIMD On | {KERNEL_SIMD} | {DELTA_KERNEL} | {TOTAL_SIMD} | {DELTA_TOTAL} |

SIMD reduces windowing+magnitude time by {SIMD_KERNEL_GAIN}% and contributes an additional {SIMD_TOTAL_GAIN}% to total speedup beyond threading alone.

### 7.4 Input-Size Sensitivity
Short (15 s), medium (45 s), and long (90 s) clips demonstrate that larger inputs amortise fixed costs (WAV parsing, setup). Speedup remains within {SPEEDUP_VARIATION}% across lengths, though shorter clips exhibit lower efficiency due to constant overheads dominating.

## Correctness & Output Quality
**Table 4. Baseline vs parallel output validation (placeholder).**

| Metric | Baseline | Parallel | Difference |
|--------|----------|----------|------------|
| Onsets detected | {BASE_ONSETS} | {PAR_ONSETS} | {DELTA_ONSETS} |
| Mean onset error (ms) | {BASE_ONSET_ERR} | {PAR_ONSET_ERR} | {DELTA_ONSET_ERR} |
| Mean pitch error (cents) | {BASE_PITCH_ERR} | {PAR_PITCH_ERR} | {DELTA_PITCH_ERR} |
| Alignment off-by-notes | {BASE_ALIGN_ERR} | {PAR_ALIGN_ERR} | {DELTA_ALIGN_ERR} |

All differences fall within the defined tolerances (±20 ms onsets, ±100 cents pitch, ±1 note alignment). Visual overlays of spectrograms and aligned note markers are indistinguishable between builds.

## Discussion
Applying Amdahl’s Law with measured data yields a serial fraction \(f_s ≈ {FSERIAL}\) and parallel fraction \(f_p = 1 - f_s\). Predicted speedup for N threads is \(S(N) = 1 / (f_s + f_p / N)\), matching observed saturation around {SATURATION_THREADS} threads. Remaining limits stem from:

- **Serial Stages:** WAV decoding, UI updates, and alignment logic that depend on sequential dependencies.
- **Memory Bandwidth:** Multithreaded FFTs stress shared caches; beyond {BANDWIDTH_THREADS} threads, bandwidth becomes the bottleneck.
- **Granularity:** Small FFT sizes reduce per-thread work, leading to overheads when N is high. Batch scheduling mitigates but cannot eliminate this.
- **UI Constraints:** WPF UI thread remains single-threaded; analysis runs in a CLI mode or background task to avoid blocking rendering.

Future gains require reducing the serial portion (e.g., parallel score alignment) and optimising memory access (structure-of-arrays layouts).

## Conclusions & Future Work
Thread-level parallelism across STFT frames combined with SIMD vectorisation delivered a {TOTAL_SPEEDUP}× improvement in end-to-end analysis time on {CPU_MODEL}. The most significant contributors were balanced frame partitioning, thread-local buffers, and vectorised magnitude computations. Correctness remained within the defined musical tolerances, validating the approach.

Future work includes offloading STFT to a GPU, reorganising spectral data into cache-friendly layouts, fusing onset and histogram kernels to reduce memory traffic, and exploring streaming pipelines that overlap decoding with analysis.

## Reproducibility Appendix
Run the CLI analysis with:

```bash
DigitalMusicAnalysis.exe --wav data/sample.wav --xml data/score.musicxml 
  --threads {N} --simd {on|off} --csv results/run_{N}.csv --repeat 3
```

- **Dataset:** Fixed WAV (30–60 s) and matching MusicXML stored under `data/`. Document duration and sample rate in the lab notebook.
- **CSV Schema:** `timestamp,commit,stage,threads,simd,duration_ms,notes,frames`.
- **Metadata:** Record CPU model, core/thread counts, OS build, .NET version, and Git commit hash for every run.

## Summary
DigitalMusicAnalysis originally processed audio sequentially, with STFT and per-frame feature extraction dominating runtime. By parallelising frame-wise kernels across threads and vectorising inner loops, the optimised build achieves {TOTAL_SPEEDUP}× speedup on {CPU_MODEL} while preserving musical correctness. Speedup tapers once serial alignment and I/O costs dominate, consistent with Amdahl’s Law. Future work targets GPU acceleration, better data layouts, and pipeline overlap to push performance further.

---

### Mini-Addendum for Codex (Optional)

**Table Templates (LaTeX/Markdown).**

```markdown
% Table 1
| Stage | Duration (ms) | % of Total |
|-------|---------------|------------|
| ... | ... | ... |

% Table 2
| Threads (N) | Total Time T_N (ms) | Speedup S(N) | Efficiency E(N) |
|-------------|---------------------|--------------|------------------|
| ... | ... | ... | ... |

% Table 3
| Configuration | Kernel Time (ms) | Δ vs Scalar | End-to-End Time (ms) | Δ vs Scalar |
|---------------|------------------|-------------|----------------------|-------------|
| ... | ... | ... | ... | ... |

% Table 4
| Metric | Baseline | Parallel | Difference |
|--------|----------|----------|------------|
| ... | ... | ... | ... |
```

**Matplotlib Script Stub.**

```python
import pandas as pd
import matplotlib.pyplot as plt

csv = pd.read_csv('results/run_summary.csv')

# Speedup vs threads
speedup = csv[csv['stage'] == 'Total'].groupby('threads')['duration_ms'].median()
threads = speedup.index.to_numpy()
T1 = speedup.loc[1]
S = T1 / speedup.values
plt.figure()
plt.plot(threads, S, marker='o')
plt.xlabel('Threads (N)')
plt.ylabel('Speedup S(N)')
plt.title('DigitalMusicAnalysis End-to-End Speedup')
plt.grid(True)
plt.savefig('fig_speedup.png', dpi=150)

# Per-stage stacked bars
stages = ['ReadWav', 'STFT', 'OnsetDetect', 'PitchEstimate', 'AlignToScore', 'Render']
T1_stage = {stage: csv[(csv['stage'] == stage) & (csv['threads'] == 1)]['duration_ms'].median() for stage in stages}
TN_stage = {stage: csv[(csv['stage'] == stage) & (csv['threads'] == threads.max())]['duration_ms'].median() for stage in stages}
plt.figure()
plt.bar(range(len(stages)), [T1_stage[s] for s in stages], label='1 thread')
plt.bar(range(len(stages)), [TN_stage[s] for s in stages], bottom=[T1_stage[s] for s in stages], label=f'{threads.max()} threads')
plt.xticks(range(len(stages)), stages, rotation=45, ha='right')
plt.ylabel('Time (ms)')
plt.title('Per-Stage Runtime Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('fig_per_stage.png', dpi=150)

# SIMD ablation
simd = csv[(csv['stage'] == 'STFT') & (csv['threads'] == threads.max())]
scalar = simd[simd['simd'] == 'off']['duration_ms'].median()
vector = simd[simd['simd'] == 'on']['duration_ms'].median()
plt.figure()
plt.bar(['Scalar', 'SIMD'], [scalar, vector], color=['#d55e00', '#0072b2'])
plt.ylabel('Kernel Time (ms)')
plt.title('STFT Kernel SIMD Ablation')
plt.savefig('fig_simd.png', dpi=150)
```
