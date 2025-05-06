# **BoundaryAware-CellSeg**  
A Boundary-Aware Cell Segmentation Approach using Region Growing_

## **Overview**  
This project focuses on segmenting cells from microscopy images using a region-growing approach while maintaining cell boundaries. The method integrates:  
- **Active Contours** (scikit-image)  
- **Anisotropic Diffusion** (Perona-Malik filtering)  
- **Gray-Weighted Distance Transform**  

Given:  
1. **Marker Image** – Small, manually or automatically detected regions within cells.  
2. **Gene Expression Image** – Grayscale intensity map showing cell walls, nuclei, and intercellular spaces.  

The goal is to expand the markers iteratively while ensuring that cells remain separated using intensity-based stopping criteria.

---

## **Installation**  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/STGK10/BoundaryAware-CellSeg.git  
cd BoundaryAware-CellSeg
pip install -r requirements.txt
```

### **Dependencies**  
- Python 3.8+  
- OpenCV  
- NumPy  
- scikit-image  
- Matplotlib
- math
- heapq
- numba import jit, prange
- os

---

## **Usage**  
### **1. Run Basic Segmentation**  
Execute the main script to segment cells:  
```bash
python src/combocellseg.py --input data/--output results/segmented.tif
```

### **2. Visualize Intermediate Steps**  
Run a Jupyter Notebook:  
```bash
jupyter notebook notebooks/combocellseg.ipynb
```

---

## **Project Structure**  
📂 **BoundaryAware-CellSeg/**  
 ├── 📂 **data/** → Sample microscopy images & markers 
 ├── 📂 **experiments/** → Logs, visualizations of results 
 ├── 📂 **models/** → (Optional) Trained models for ML-based segmentation  
 ├── 📂 **notebooks/** → Jupyter notebooks for testing segmentation  
 │    ├── `activecontour.ipynb` → Active contour-based refinement  
 │    ├── `peronamalika.ipynb` → Preprocessing with Perona-Malik diffusion  
 │    ├── `gwdt.ipynb` → Distance transform-based segmentation  
 │    ├── `combocellseg.ipynb` → Implementation of boundary-aware cells segmentation with visualization
 ├── 📂 **src/** → Core algorithms 
 │    ├── `combocellseg.py` → full Implementation of boundary-aware combining peronam-malik, gwdt and active contour for cells segmentation 
 ├── 📜 `requirements.txt` → Dependency list  
 ├── 📜 `README.md` → Project documentation  
  

---

## **References**  
- Yokoi, S., Toriwaki, J.-I. & Fukumura, T. (1981). "On Generalized Distance Transformation of Digitized Pictures." IEEE PAMI.  
- Wang, Y., Wang, C. & Zhang, Z. (2018). "Segmentation of clustered cells in negative phase contrast images." Journal of Microscopy.  
- Perona, P. & Malik, J. (1990). "Scale-space and edge detection using anisotropic diffusion." IEEE PAMI.
- Active Contours Without Edges. Tony F. Chan, Member, IEEE, and Luminita A. Vese.

---

## **License**  
MIT License  
