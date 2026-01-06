# Part 3 (Run #3) Experimental Results Figures

This directory contains publication-quality figures for the Run #3 section of your report.

## Generated Figures

### Figure 1: Feature Comparison (`fig1_feature_comparison.png/pdf`)
**Purpose:** Compare different feature extraction methods
- **(a)** Horizontal bar chart showing accuracy of each method
- **(b)** Scatter plot showing the trade-off between feature dimensionality and performance

**Key Insights:**
- PHOW achieves the highest accuracy (70%)
- Higher dimensionality doesn't always mean better performance
- GIST features work better with ensemble methods (Random Forest)

**Recommended Caption:**
```
Figure 1: Comparison of feature extraction methods. (a) Accuracy comparison across different 
feature types. (b) Trade-off between feature dimensionality and classification accuracy. 
PHOW achieves the best performance despite not having the highest dimensionality.
```

---

### Figure 2: Hyperparameter Tuning (`fig2_hyperparameter_tuning.png/pdf`)
**Purpose:** Show the effect of regularization parameter C on different methods
- **(a)** PHOW + Chi2 Kernel + LinearSVC
- **(b)** Dense SIFT Pyramid + Chi2 Kernel + LinearSVC
- **(c)** GIST + LinearSVC

**Key Insights:**
- Optimal C value varies by feature type
- PHOW is relatively stable across C values
- GIST performance increases with higher C (needs more regularization flexibility)

**Recommended Caption:**
```
Figure 2: Hyperparameter tuning results showing the effect of regularization parameter C 
on classification accuracy for three different feature methods. Each method exhibits 
different optimal C values, with PHOW showing best performance at C=0.1.
```

---

### Figure 3: Classifier Comparison (`fig3_classifier_comparison.png/pdf`)
**Purpose:** Compare different classifiers across various features
- Shows KNN, LinearSVC, Chi2+LinearSVC, and Random Forest
- Grouped by feature type

**Key Insights:**
- Chi2 Kernel Map significantly improves LinearSVC performance
- Different features prefer different classifiers
- PHOW benefits most from Chi2 kernel approximation

**Recommended Caption:**
```
Figure 3: Classifier performance comparison across different feature types. Chi2 kernel 
approximation with LinearSVC consistently outperforms other classifiers for BoVW-based 
features, while Random Forest works well with global features like GIST.
```

---

### Figure 4: Pyramid Analysis (`fig4_pyramid_analysis.png/pdf`)
**Purpose:** Analyze the impact of pyramid structures
- **(a)** Spatial pyramid levels vs accuracy (PHOW)
- **(b)** Gaussian pyramid scales vs accuracy

**Key Insights:**
- More spatial levels improve accuracy but with diminishing returns
- Spatial information is more valuable than multi-scale for scene recognition
- Feature dimensionality increases rapidly with spatial levels

**Recommended Caption:**
```
Figure 4: Impact of pyramid structures on classification performance. (a) Spatial pyramid 
levels in PHOW show significant accuracy improvements with finer spatial divisions. 
(b) Gaussian pyramid scales show moderate improvements, with diminishing returns after 3 scales.
```

---

### Figure 5: Feature Fusion (`fig5_feature_fusion.png/pdf`)
**Purpose:** Demonstrate the effects of combining features
- Compares single features vs combined features
- Shows accuracy and dimensionality trade-offs

**Key Insights:**
- Feature fusion can improve performance but not always
- PHOW+GIST slightly reduces accuracy due to feature dominance
- Combined methods offer good dimensionality-performance balance

**Recommended Caption:**
```
Figure 5: Feature fusion analysis comparing individual and combined feature methods. 
While PHOW-Gaussian achieves similar accuracy to PHOW alone with reduced dimensionality, 
simple concatenation (PHOW+GIST) shows modest performance due to feature imbalance.
```

---

### Figure 6: Architecture Diagram (`fig6_architecture_diagram.png/pdf`)
**Purpose:** Visual explanation of different pyramid architectures
- **(a)** PHOW (Spatial pyramid only)
- **(b)** Dense SIFT Pyramid (Gaussian pyramid only)
- **(c)** PHOW-Gaussian (Combined approach)

**Key Insights:**
- Spatial pyramid captures "where" information
- Gaussian pyramid captures "at what scale" information
- Combined approach leverages both types of information

**Recommended Caption:**
```
Figure 6: Architectural comparison of pyramid-based feature extraction methods. 
(a) PHOW uses spatial pyramid pooling to preserve spatial layout. (b) Dense SIFT 
Pyramid uses Gaussian pyramid for multi-scale representation. (c) PHOW-Gaussian 
combines both approaches for richer feature representation.
```

---

### Figure 7: Summary Table (`fig7_summary_table.png/pdf`)
**Purpose:** Comprehensive comparison of all methods
- Lists all implemented methods
- Shows feature dimensions, spatial/multi-scale properties, accuracy, and best parameters

**Key Insights:**
- Complete overview of experimental results
- Easy comparison of method characteristics
- Color-coded by performance

**Recommended Caption:**
```
Table 1: Summary of Run #3 methods and experimental results. Methods are compared 
across feature dimensionality, spatial pooling capability, multi-scale properties, 
and classification accuracy. Colors indicate performance levels (green=best, red=lowest).
```

---

## Usage in Report

### Recommended Figure Selection

For a **4-page report**, we recommend including:
1. **Figure 1** - Essential for showing overall method comparison
2. **Figure 2 or 4** - Show either hyperparameter tuning OR pyramid analysis
3. **Figure 6** - Visual architecture comparison is very effective
4. **Figure 7** - Summary table provides complete overview

### LaTeX Integration

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/fig1_feature_comparison.pdf}
    \caption{Your caption here}
    \label{fig:feature_comparison}
\end{figure}
```

### File Formats
- **PNG files** (300 DPI): For direct viewing and MS Word
- **PDF files** (vector): For LaTeX documents (preferred)

---

## Regenerating Figures

To regenerate all figures with updated data:

```bash
python plot_part3_results.py
```

All figures will be saved in this directory.

---

## Key Statistics for Report

### Best Results:
- **Best overall accuracy:** 70.00% (PHOW + Chi2-LinearSVC, C=0.1)
- **Best feature efficiency:** Dense SIFT Pyramid (750 dims, 65.33% accuracy)
- **Best combined method:** PHOW-Gaussian (2500 dims, ~68% accuracy)

### Performance Rankings:
1. PHOW (Spatial): 70.00%
2. PHOW+GIST: 68.67%
3. PHOW-Gaussian: 68.00%
4. Dense SIFT Pyramid: 65.33%
5. GIST+RF: 57.33%

### Computational Efficiency:
- GIST: Fastest (512 dims)
- Dense SIFT Pyramid: Moderate (750 dims)
- PHOW-Gaussian: Good balance (2500 dims)
- PHOW: Most expensive (5250 dims)

---

## Notes

- All figures use consistent color schemes and styling
- Both PNG (raster) and PDF (vector) formats are provided
- Figures are designed to be publication-quality (300 DPI)
- Color schemes are colorblind-friendly where possible


