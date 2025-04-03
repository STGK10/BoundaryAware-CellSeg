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

---

## **Usage**  
### **1. Run Basic Segmentation**  
Execute the main script to segment cells:  
```bash
python src/regrow.py --input data/image1.tif --markers data/image2.tif --output results/segmented.tif
```

### **2. Visualize Intermediate Steps**  
Run a Jupyter Notebook:  
```bash
jupyter notebook notebooks/Segdemo.ipynb
```

---

## **Project Structure**  
📂 **BoundaryAware-CellSeg/**  
 ├── 📂 **notebooks/** → Jupyter notebooks for testing segmentation  
 ├── 📂 **src/** → Core algorithms  
 │    ├── `regrow.py` → Implements boundary-aware region growing  
 │    ├── `actcont.py` → Active contour-based refinement  
 │    ├── `peronamalika.py` → Preprocessing with Perona-Malik diffusion  
 │    ├── `gwdt.py` → Distance transform-based segmentation  
 │    ├── `utils.py` → Helper functions  
 ├── 📂 **data/** → Sample microscopy images & markers  
 ├── 📂 **experiments/** → Logs, visualizations of results  
 ├── 📂 **models/** → (Optional) Trained models for ML-based segmentation  
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
