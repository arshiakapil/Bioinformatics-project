# CNN-Based CNV Detection — Implementation Plan

**Target file:** `cnn_cnv/cnn_cnv_analysis.ipynb`
**Working directory:** `/Users/arshkapil/Downloads/Bioinformatics-project/cnn_cnv/`

---

## Overview

A self-contained Jupyter notebook that simulates read-depth data for human
chromosome 22, normalises it three ways, trains a 1-D CNN to classify genomic
windows as **deletion / normal / duplication**, and produces three publication-
quality figures plus a printed summary report.

All analysis is pure Python (numpy / pandas / scipy / matplotlib / tensorflow or
torch).  No external aligners or reference downloads are required.

---

## Cell-by-Cell Structure (target: ~30 cells)

### Block 1 — Setup (cells 1–3)

| Cell | Type | Content |
|------|------|---------|
| 1 | Markdown | Title, author, date, project description |
| 2 | Code | Imports: numpy, pandas, scipy, matplotlib, seaborn, sklearn, tensorflow/keras (or torch) |
| 3 | Code | Global constants: chromosome length, window size, CNV parameters, random seed |

**Key constants**
```
CHROM_LEN   = 51_304_566   # chr22 hg38 length (bp)
WINDOW_SIZE = 1_000        # 1 kb bins
N_WINDOWS   = CHROM_LEN // WINDOW_SIZE   # ~51 304 bins
MEAN_DEPTH  = 30           # target average coverage
CNV_FRAC    = 0.08         # 8 % of windows carry a CNV
SEED        = 42
```

---

### Block 2 — Data Simulation (cells 4–8)

#### Cell 4 — Chromosome 22 genome model
- Create a pandas DataFrame with columns:
  `window_id`, `chrom_start`, `chrom_end`, `gc_content`, `mappability`
- GC content: sinusoidal baseline (~0.48 mean) with Gaussian noise, mirroring
  the known GC profile of chr22 (high-GC pericentromeric → lower-GC telomeric).
- Mappability: uniform 0–1 score with low-mappability troughs at repeat-rich
  regions (simulate 3 low-map zones of width ~200 windows each).

#### Cell 5 — CNV placement
- Randomly assign CNV labels to `CNV_FRAC * N_WINDOWS` windows:
  - **Deletion** (copy number 1): depth multiplier ~ 0.5
  - **Duplication** (copy number 3): depth multiplier ~ 1.5
  - **Normal** (copy number 2): multiplier = 1.0
- CNVs are placed in contiguous blocks of 5–50 windows (realistic segment size).
- Store ground-truth labels in column `true_label` (0 = deletion, 1 = normal,
  2 = duplication).

#### Cell 6 — Raw read-depth generation
- Base depth per window: `Poisson(MEAN_DEPTH × gc_bias × mappability_bias)`
- Apply CNV multiplier to relevant windows.
- Add three sources of technical noise:
  1. **GC bias**: depth scales with `exp(β_gc × (gc – 0.5))`, β_gc ~ 1.2
  2. **Mappability bias**: depth scales linearly with mappability score
  3. **Batch effect**: assign windows to 4 batches; add per-batch depth offset
     drawn from `Normal(0, 3)`
- Store raw depth in column `raw_depth`.

#### Cell 7 — Train / test split
- Split windows 70 / 15 / 15 (train / val / test) stratified by `true_label`.
- Save split indices; use them consistently through all downstream steps.

#### Cell 8 — Markdown summary table
- Print: total windows, CNV counts per class, mean/std raw depth per class.

---

### Block 3 — Normalisation (cells 9–15)

Three independent normalisation methods are applied to the same raw depth signal.
Each produces a new column in the DataFrame and a normalised feature matrix used
by the CNN.

#### Method 1 — GC-content correction (cells 9–10)
**Algorithm:** LOESS / rolling-median regression of depth on GC content.
1. Bin windows by GC decile.
2. Compute median depth per decile on the training set.
3. Fit a degree-2 polynomial to (gc_decile_centre, median_depth).
4. Divide each window's raw depth by the predicted GC-median depth, then
   multiply by the global median depth.
5. Store result in `gc_norm_depth`.

#### Method 2 — Mappability-weighted normalisation (cells 11–12)
**Algorithm:** Inverse-mappability scaling followed by z-score standardisation.
1. Divide raw depth by mappability score (floor mappability at 0.1 to avoid
   division by near-zero).
2. Z-score the result using training-set mean and std.
3. Store result in `map_norm_depth`.

#### Method 3 — Median-ratio batch normalisation (cells 13–14)
**Algorithm:** Same median-ratio method used in DESeq2-style normalisation.
1. Compute a pseudo-reference depth per window = geometric mean of depths
   across the 4 batches.
2. For each batch, compute the median of (batch_depth / pseudo_reference).
3. Divide each window's depth by its batch size factor.
4. Store result in `batch_norm_depth`.

#### Cell 15 — Normalisation comparison
- Print a table: per-class mean ± std for raw, gc_norm, map_norm, batch_norm.
- Show coefficient of variation (CV) reduction across methods.

---

### Block 4 — Feature Engineering (cells 16–17)

#### Cell 16 — Window feature matrix
For each window, build a feature vector of length **W = 15** centred on that
window (±7 neighbours) using whichever normalisation is used as input.  This
gives the CNN local context without explicit recurrence.

- Shape: `(N_WINDOWS, W, 3)` — last axis = [gc_norm, map_norm, batch_norm]
- Pad edges with zeros.
- Store as numpy arrays `X_train`, `X_val`, `X_test`.
- One-hot encode labels into `y_train`, `y_val`, `y_test`.

#### Cell 17 — Markdown: feature engineering summary
- Describe input tensor shape and label encoding.

---

### Block 5 — CNN Model (cells 18–22)

#### Cell 18 — Architecture definition
A lightweight 1-D CNN appropriate for genomic window classification:

```
Input: (15, 3)
→ Conv1D(32, kernel=3, activation='relu', padding='same')
→ BatchNormalization
→ Conv1D(64, kernel=3, activation='relu', padding='same')
→ BatchNormalization
→ GlobalAveragePooling1D
→ Dense(64, activation='relu')
→ Dropout(0.3)
→ Dense(3, activation='softmax')   # deletion / normal / duplication
```

- Optimizer: Adam (lr = 1e-3)
- Loss: categorical cross-entropy
- Metrics: accuracy, macro-F1 (custom or sklearn callback)

#### Cell 19 — Training
- Epochs: 50, batch size: 256
- Callbacks: EarlyStopping (patience=8, restore_best_weights=True),
  ReduceLROnPlateau (factor=0.5, patience=4)
- Save training history to a dict.

#### Cell 20 — Evaluation on test set
- Compute: accuracy, precision, recall, F1 (per class + macro/weighted)
- Print sklearn `classification_report`.
- Compute and display confusion matrix values.

#### Cell 21 — ROC / AUC (one-vs-rest)
- Compute AUC for each class.
- Print a summary table.

#### Cell 22 — Markdown: model summary
- Re-state architecture, parameter count, final test metrics.

---

### Block 6 — Visualisations (cells 23–25)

#### Cell 23 — Figure 1: Chromosome 22 Read-Depth Profile
A 2-row figure (before / after normalisation):
- **Top panel:** Raw depth along chr22 x-axis (window index), coloured by true
  label (deletion=red, normal=grey, duplication=blue).  Shade low-mappability
  zones.
- **Bottom panel:** Batch-normalised depth with the same colouring.
- Add a secondary y-axis showing GC content as a line overlay on both panels.
- Save as `viz1_depth_profile.png`.

#### Cell 24 — Figure 2: Normalisation Method Comparison
A 3-column violin + box plot figure:
- One column per normalisation method (gc_norm, map_norm, batch_norm).
- Each column has 3 violins: deletion, normal, duplication.
- Annotate with per-class CV and inter-class separation (Cohen's d: normal vs
  deletion, normal vs duplication).
- Save as `viz2_normalisation_comparison.png`.

#### Cell 25 — Figure 3: CNN Performance Dashboard
A 4-panel figure:
- **Panel A:** Training vs validation loss curves (epochs on x-axis).
- **Panel B:** Training vs validation accuracy curves.
- **Panel C:** Confusion matrix heatmap (test set, row-normalised to show
  recall per class).
- **Panel D:** Per-class precision / recall / F1 grouped bar chart.
- Save as `viz3_cnn_performance.png`.

---

### Block 7 — Summary Report (cells 26–28)

#### Cell 26 — Collect metrics
Aggregate all key numbers into a results dict:
- Simulation: N windows, N CNVs, mean depth, GC range
- Normalisation: CV per method per class
- Model: accuracy, macro-F1, per-class AUC, confusion matrix

#### Cell 27 — Print report
```
╔══════════════════════════════════════════════════════════╗
║       CNN-BASED CNV DETECTION — SUMMARY REPORT          ║
╠══════════════════════════════════════════════════════════╣
║  SIMULATION                                              ║
║    Chromosome       : 22 (hg38, 51,304,566 bp)          ║
║    Window size      : 1,000 bp  |  Total windows: 51,304 ║
║    Mean raw depth   : 30×       |  CNV rate: 8%          ║
║    Deletions        : XXX  |  Duplications: XXX          ║
╠══════════════════════════════════════════════════════════╣
║  NORMALISATION                                           ║
║    GC correction CV reduction   : XX% → XX%             ║
║    Mappability norm CV reduction : XX% → XX%             ║
║    Batch norm CV reduction       : XX% → XX%             ║
╠══════════════════════════════════════════════════════════╣
║  CNN MODEL                                               ║
║    Parameters       : ~XX,XXX                           ║
║    Test accuracy    : XX.X%                             ║
║    Macro F1         : X.XXX                             ║
║    Deletion AUC     : X.XXX                             ║
║    Normal AUC       : X.XXX                             ║
║    Duplication AUC  : X.XXX                             ║
╠══════════════════════════════════════════════════════════╣
║  QC CHECKLIST                                            ║
║    [✓/✗] Test accuracy ≥ 0.90                           ║
║    [✓/✗] Macro F1 ≥ 0.88                                ║
║    [✓/✗] All class AUC ≥ 0.92                           ║
║    [✓/✗] Batch CV reduced by ≥ 30% after normalisation  ║
╚══════════════════════════════════════════════════════════╝
```

#### Cell 28 — Markdown: conclusion
- Interpret results in context of CNV detection sensitivity.
- Note limitations (simulated data, no allele frequency, no SV breakpoints).
- Suggest next steps (real WGS data, breakpoint detection, ensemble methods).

---

## File Outputs

| File | Description |
|------|-------------|
| `cnn_cnv_analysis.ipynb` | Main notebook (~30 cells, fully executed) |
| `viz1_depth_profile.png` | Chr22 depth profile before/after normalisation |
| `viz2_normalisation_comparison.png` | Violin plots comparing 3 norm methods |
| `viz3_cnn_performance.png` | CNN training curves + confusion matrix + F1 bars |

---

## Dependencies

```python
numpy >= 1.24
pandas >= 2.0
scipy >= 1.11
matplotlib >= 3.7
seaborn >= 0.12
scikit-learn >= 1.3
tensorflow >= 2.13   # or torch >= 2.0 with equivalent API
```

---

## Implementation Notes

- Use `np.random.default_rng(SEED)` throughout for reproducibility.
- All normalisation fit parameters are derived from training windows only and
  applied to val/test to avoid data leakage.
- CNV block sizes are drawn from `randint(5, 50)` to mimic realistic segment
  lengths of 5–50 kb.
- The CNN input uses all three normalised channels simultaneously so the model
  implicitly learns which normalisation is most informative per region.
- Target runtime: < 5 minutes on CPU.
