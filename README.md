# Characterization and Classification of Mitochondrial Structures

This project uses texture-based image analysis to classify mitochondrial structures (rods, puncta, fibers) from microscopy images. It includes scripts for texture analysis based feature extraction using the pyRadiomics library, and feature selection and machine learning classification using python.

------------------------------------------------------------------------

## ⚙️ Requirements

-   Python 3.9+
-   Python packages:
    -   numpy\>=1.9.2
    -   pandas
    -   SimpleITK
    -   Pillow
    -   radiomics
    -   scikit-learn
    -   scikit-image
    -   matplotlib
    -   xlsxwriter
    -   tk
    -   ttkbootstrap

Install dependencies with:

``` bash
pip install -r requirements.txt
```

## 🏁 Getting Started

This tool is designed for biologists to extract quantitative features relating to image texture from microscopy images and perform machine learning classification. You do **not** need to be a Python expert to use it, but basic familiarity with running scripts is helpful.

### Python environment
* We recomment installing [Anaconda](https://www.anaconda.com/) to manage Python and the dependencies
* Once installed, create a new environment and install the required packages:

``` bash
conda create -n mito_env python=3.9
conda activate mito_env
pip install -r requirements.txt
```
### Preparing your images
* Pre-process your images as necessary (i.e., denoising)
* Save images as ```.tiff ```

### Launch the Graphical User Interface (GUI)
* Open your command line in the folder containing the script, then run:

``` bash
python Mito_feature_extraction_gui.py
```
### Feature extraction
1. Select **input** and **output** directories
2. Choose the texture group to extract (FOS, GLCM, GLRLM, GLDM, GLSZM, NGTDM)
3. The GUI will save the results as both ```.csv``` and ```.xlsx``` files

### Recursive Feature Elimination (RFE)
1. Select your labelled ```.csv``` file
2. Choose an **output directory** for the results
3. The RFE selects the top 20 features

### Machine Learning Classification
1. Select the model
    * Decision Tree 
    * Support Vector Machine (SVM) 
2. Choose the data type 
    * Binary
    * Multiclass
3. Select the RFE-feature set as the input
4. Run the machine learning pipeline to generate classification results (confusion matrix, ROC, and classification report)
------------------------------------------------------------------------

## Features extracted with texture analysis

The feature extraction methods used were implemented in the paper:

[Joost J. M. van Griethuysen, Andriy Fedorov, Chintan Parmar, Ahmed Hosny, Nicolas Aucoin, Vivek Narayan, Regina G. H. Beets-Tan, Jean-Christophe Fillion-Robin, Steve Pieper, and Hugo J. W. L. Aerts. Computational radiomics system to decode the radiographic phenotype. Cancer Research, 77(21): e104–e107, 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5672828/)

The extracted features come from a family of texture analysis techniques and first order statistics.

## First Order Statistics and Texture analysis

The following subsections will briefly outline what each texture family and first order statistics features extract.

### First Order Statistics (FOS)

FOS extracts features relating to the pixel intensities themselves, and does not provide spatial information [1,2]. Some features of note as mean, skewness, kurtosis and standard deviation.

### Gray Level Co-Occurrence Matrix (GLCM)

GLCM is based on how often a pair of pixel intensities are found next to each other [3,4]. GLCM is directionally dependent providing spatial organization.

### Gray Level Run Length Matrix (GLRLM)

GLRLM investigates the gray level run lengths in an image; a run length is when there is a consecutive length of pixels that have the same gray level intensities and are collinearly organized [5]. GLRLM is directionally dependent providing insights into the level of fragmentation.

### Gray Level Dependence Matrix (GLDM)

GLDM quantifies the gray level dependencies, a gray level dependency refers to the number of connected voxels within a distance [6]. These voxels are dependent on a central point [6]. GLDM is directionally dependent, and can be used to study textural uniformity and organization.

### Gray Level Size Zone Matrix (GLSZM)

GLSZM provides insights into texture complexity by quantifying the size of a zone, a zone is the number of connected voxels that share the same gray level intensity [7]. Unlike the previously outlined texture groups this is not directionally dependent.

### Neighbouring Gray Tone Difference Matrix (NGTDM)

NGTDM quantifies the differences between the gray level intensities and the average intensity of a neighbourhood of pixels within a specified distance, thus providing insights into finer textural differences (i.e., coarseness) [8].

---
## References
[1] Nidhi Aggarwal and Rajendra Kumar Agrawal. First and second order statistics features for classification of magnetic resonance brain images. Journal of Signal Processing Systems, 3:146–153, 2012.

[2] M. Bevk and I. Kononenko. A statistical approach to texture description of medical images: a preliminary study. In Proceedings of 15th IEEE Symposium on Computer-Based Medical Systems (CBMS 2002), pages 239–244, 2002.

[3] Robert M. Haralick, K. Shanmugam, and Its’Hak Dinstein. Textural featuresfor image classification. IEEE Transactions on Systems, Man, and Cybernetics, SMC-3(6):610–621, 1973.

[4] Tommy L ̈ofstedt, Patrik Brynolfsson, Thomas Asklund, Tufve Nyholm, and Anders Garpebring. Gray-level invariant haralick texture features. PLoS One, 14(2):e0212110, February 2019.

[5] Mary M. Galloway. Texture analysis using gray level run lengths. Computer Graphics and Image Processing, 4(2):172–179, 1975.

[6] Chengjun Sun and William G Wee. Neighboring gray level dependence matrix for texture classification. Computer Vision, Graphics, and Image Processing, 23(3):341–352, 1983.

[7] Guillaume Thibault, Jesus Angulo, and Fernand Meyer. Advanced statistical matrices for texture characterization: Application to cell classification. IEEE Transactions on Biomedical Engineering, 61(3):630–637, 2014.

[8] M. Amadasun and R. King. Textural features corresponding to textural properties. IEEE Transactions on Systems, Man, and Cybernetics, 19(5):1264–1274, 1989.

---

## Acknowledgements

-   Developed as part of Amulya Kaianathbhatta's Masters dissertation at Carleton University. DOI: <https://doi.org/10.22215/etd/2024-16247>
-   Supervised by Dr. Leila Mostaço-Guidolin in the [Tissue Engineering and BioImaging Lab](https://www.teb-lab.com/)
-   Texture analysis- [PyRadiomics](https://pyradiomics.readthedocs.io/en/latest/)
-   All machine learning and RFE filter based feature selection was implemented by Natasha Kunchur
