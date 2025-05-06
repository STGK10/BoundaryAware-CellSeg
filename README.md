# 🧬 **BoundaryAware-CellSeg**

**A Boundary-Aware Cell Segmentation Method Using Region Growing and Intensity-Based Constraints**

---

## 🔍 Overview

**BoundaryAware-CellSeg** is a Python-based toolkit for cell segmentation in microscopy images. The core algorithm uses a region-growing strategy to expand seed markers into full cell regions while preserving accurate cell boundaries based on intensity information.

It combines three techniques:

* **Active Contours** (via *scikit-image*)
* **Anisotropic Diffusion Filtering** (*Perona-Malik*)
* **Gray-Weighted Distance Transform** (*GWDT*)

**🧠 Key innovation: SIEMO's Stopping Criterion for Image Segmentation**

This project introduces **SIEMO's Stopping Criterion**, a powerful multi-condition strategy to halt the segmentation process reliably:

1. **Level Set Function (LSF) Convergence**  
   $\| \phi^{(k+1)} - \phi^{(k)} \|_2 < \delta$  
   *(Stops when the evolving curve stabilizes)*

2. **Change in Segmented Area**  
   $\left| A^{(k+1)} - A^{(k)} \right| < \tau$  
   *(Ensures the segmented area does not oscillate)*

3. **Intersection-Based Early Stop**  
   $R^{(k+1)} \cap P \neq \emptyset$  
   *(Prevents redundant segmentation by stopping if overlap with a known region is detected)*

These conditions are **combined** as:

$$
\left( \| \phi^{(k+1)} - \phi^{(k)} \|_2 < \delta \ \wedge\ \left| A^{(k+1)} - A^{(k)} \right| < \tau \right) \ \vee \ \left( R^{(k+1)} \cap P \neq \emptyset \right)
$$

✅ This adaptive rule enables precise and efficient segmentation — balancing quality, speed, and safety.



### 📘 Input

1. **Marker Image** — Seed points inside cells (manual or automatic).
2. **Gene Expression Image** — Grayscale image showing cell walls, nuclei, and boundaries.

The algorithm expands the seed regions iteratively using local intensity and gradient information to prevent merging across cell borders.

---

## ⚙️ Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/STGK10/BoundaryAware-CellSeg.git
cd BoundaryAware-CellSeg
pip install -r requirements.txt
```

### ✅ Dependencies

* Python 3.8+
* OpenCV
* NumPy
* scikit-image
* Matplotlib
* Numba
* Standard libraries: `os`, `math`, `heapq`

---

## 🚀 Usage

### 🔹 Run Basic Segmentation

To apply boundary-aware segmentation on an image:

```bash
python src/combocellseg.py --input data/ --output segmented_result/
```

### 🔹 Visualize Intermediate Steps

For step-by-step execution and visualization, use the Jupyter notebook:

```bash
jupyter notebook notebooks/combocellseg.ipynb
```

---

## 🗂️ Project Structure

```
BoundaryAware-CellSeg/
├── data/               # Sample microscopy images and marker files
├── experiments/        # Logs and result visualizations
├── models/             # ML models for advanced segmentation
├── notebooks/          # Interactive notebooks for exploring methods
│   ├── activecontour.ipynb
│   ├── peronamalika.ipynb
│   ├── gwdt.ipynb
│   └── combocellseg.ipynb
├── src/                # Core implementation
│   └── combocellseg.py
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 📚 References

* Chan, T.F. & Vese, L.A. (2001). *Active Contours Without Edges*, IEEE Trans. Image Processing.
* Perona, P. & Malik, J. (1990). *Scale-Space and Edge Detection Using Anisotropic Diffusion*, IEEE PAMI.
* Yokoi, S., Toriwaki, J.-I., & Fukumura, T. (1981). *On Generalized Distance Transformation of Digitized Pictures*, IEEE PAMI.
* Wang, Y., Wang, C., & Zhang, Z. (2018). *Segmentation of Clustered Cells in Negative Phase Contrast Images*, Journal of Microscopy.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---
