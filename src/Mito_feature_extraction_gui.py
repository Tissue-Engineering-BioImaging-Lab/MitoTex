import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import os
import json
import logging
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import pandas as pd
import numpy as np
import threading
import queue
import cv2
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter.font as tkFont

# ML imports: functions from the script machine_learning_code.py
from machine_learning_code import (
    decisiontree_binary,
    decisiontree_multiclass,
    svm_binary,
    svm_multiclass
)
from RFE_feature_selection import run_rfe_feature_selection  # Custom RFE script

# =============================
# Helper functions
# =============================
def rename_columns(col):
    """Clean up column names from Pyradiomics output for readability."""
    return col.replace('original_', '') if col.startswith('original_') else col

# Create a binary mask using Otsu thresholding
def create_thresholded_mask(image_np):
    """
    Convert raw microscopy image to a binary mask using Otsu thresholding.
    
    Parameters
    ----------
    image_np : np.ndarray
        Numpy array of a single channel image (e.g., mitochondria).
    
    Returns
    -------
    mask : np.ndarray
        Binary mask where foreground pixels correspond to structures of interest.
    """
    if image_np.dtype == np.uint16:
        image_8bit = (image_np / 256).astype(np.uint8)
    elif image_np.dtype == np.uint8:
        image_8bit = image_np
    else:
        raise ValueError("Unsupported image bit depth. Only 8-bit and 16-bit images are supported.")
    if image_8bit.ndim == 3:
        image_8bit = image_8bit[:, :, 0]
    _, mask = cv2.threshold(image_8bit, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask.astype(np.uint8)

def process_image_file(image_path, extractor):
    """
    Extract radiomic features from a single biological image.
    
    Parameters
    ----------
    image_path : str
        Path to a TIFF microscopy image.
    extractor : radiomics.featureextractor.RadiomicsFeatureExtractor
        Configured Pyradiomics feature extractor.
    
    Returns
    -------
    dict
        Extracted features along with image name, ready for tabular output.
    """
    try:
        image = Image.open(image_path)
        image_np = np.array(image)
        if image_np.ndim == 3:
            image_np = image_np[:, :, 0]
        mask_np = create_thresholded_mask(image_np)
        image_sitk = sitk.GetImageFromArray(image_np)
        mask_sitk = sitk.GetImageFromArray(mask_np)
        mask_sitk.CopyInformation(image_sitk)
        featureVector = extractor.execute(image_sitk, mask_sitk)
        if 'original_glcm_SumAverage' in featureVector and 'original_glcm_JointAverage' in featureVector:
            featureVector['original_glcm_SumAverage'] = 2 * featureVector['original_glcm_JointAverage']
        data = {'Image Name': [os.path.basename(image_path)]}
        for feature_class in ['firstorder', 'glcm', 'glrlm', 'glszm', 'ngtdm', 'gldm']:
            for key, value in featureVector.items():
                if key.startswith(f'original_{feature_class}_'):
                    data[key] = [value]
        return data
    except Exception as e:
        logging.error(f"Error processing image '{image_path}': {str(e)}")
        return None

# =============================
# Main GUI application
# =============================
class RadiomicsApp:
    """
    Graphical user interface (GUI) for radiomics feature extraction and ML classification
    of microscopy images of the mitochondria
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Mitochondria Texture Analysis User Interface")
        self.root.geometry("1200x860")
        self.root.minsize(900, 700)

        # ---------------------------
        # Variables
        # ---------------------------
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.techniques = {
            'firstorder': tk.BooleanVar(value=True),
            'glcm': tk.BooleanVar(value=True),
            'glrlm': tk.BooleanVar(value=True),
            'glszm': tk.BooleanVar(value=True),
            'ngtdm': tk.BooleanVar(value=True),
            'gldm': tk.BooleanVar(value=True)
        }
        self.classifier_var = tk.StringVar(value="decision_tree")
        self.data_type_var = tk.StringVar(value="binary")

        # ---------------------------
        # Main container frame
        # ---------------------------
        main_frame = ttk.Frame(root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=6)
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        # Configure rows and columns for expansion
        for i in range(20):
            main_frame.rowconfigure(i, weight=0)
        main_frame.rowconfigure(6, weight=1)  # Log area expands
        main_frame.rowconfigure(8, weight=1)  # ML report expands
        main_frame.columnconfigure(0, weight=1)

        # ---------------------------
        # Title
        # ---------------------------
        title_lbl = ttk.Label(main_frame, text="MitoTex",
                              font=("Segoe UI", 18, "bold"), foreground="white")
        title_lbl.grid(row=0, column=0, sticky="w", pady=(6, 12))

        # ---------------------------
        # File selection
        # ---------------------------
        file_frame = ttk.Labelframe(main_frame, text=" File Selection ", bootstyle="secondary")
        file_frame.grid(row=1, column=0, sticky="ew", padx=6, pady=6)
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="Input Folder:", width=15, foreground="white").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(file_frame, textvariable=self.input_dir).grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(file_frame, text="Browse", command=self.browse_input, bootstyle="primary-outline").grid(row=0, column=2, padx=4, pady=2)

        ttk.Label(file_frame, text="Output Folder:", width=15, foreground="white").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(file_frame, textvariable=self.output_dir).grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(file_frame, text="Browse", command=self.browse_output, bootstyle="primary-outline").grid(row=1, column=2, padx=4, pady=2)

        # ---------------------------
        # Texture techniques
        # ---------------------------
        tech_frame = ttk.Labelframe(main_frame, text=" Texture Techniques ", bootstyle="secondary")
        tech_frame.grid(row=2, column=0, sticky="ew", padx=6, pady=6)
        for idx, (tech, var) in enumerate(self.techniques.items()):
            ttk.Checkbutton(tech_frame, text=tech.upper(), variable=var, bootstyle="info").grid(row=0, column=idx, padx=6, pady=4)

        # ---------------------------
        # Run buttons
        # ---------------------------
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=3, column=0, sticky="ew", pady=6)
        btn_frame.columnconfigure((0, 1), weight=1)
        ttk.Button(btn_frame, text="Run Analysis", command=self.run_analysis_thread, bootstyle="success").grid(row=0, column=0, padx=6)
        ttk.Button(btn_frame, text="Run RFE Filter", command=self.handle_rfe_filter, bootstyle="warning").grid(row=0, column=1, padx=6)

        # ---------------------------
        # Progress bar + label
        # ---------------------------
        prog_frame = ttk.Frame(main_frame)
        prog_frame.grid(row=4, column=0, sticky="ew", pady=6)
        prog_frame.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(prog_frame, orient='horizontal', mode='determinate')
        self.progress.grid(row=0, column=0, sticky="ew", padx=4)
        self.progress_label = ttk.Label(prog_frame, text="Progress: 0%", foreground="white")
        self.progress_label.grid(row=0, column=1, sticky="w", padx=6)

        # ---------------------------
        # Log area
        # ---------------------------
        log_frame = ttk.Labelframe(main_frame, text=" Log ", bootstyle="secondary")
        log_frame.grid(row=6, column=0, sticky="nsew", padx=6, pady=6)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, bg="#111214", fg="white", insertbackground="white")
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # ---------------------------
        # ML options
        # ---------------------------
        ml_frame = ttk.Labelframe(main_frame, text=" Machine Learning ", bootstyle="secondary")
        ml_frame.grid(row=7, column=0, sticky="ew", padx=6, pady=6)

        classifier_frame = ttk.Frame(ml_frame)
        classifier_frame.grid(row=0, column=0, sticky="w", pady=2)
        ttk.Label(classifier_frame, text="Classifier:", width=12, foreground="white").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(classifier_frame, text="SVM", variable=self.classifier_var, value="svm", bootstyle="info").grid(row=0, column=1, padx=4)
        ttk.Radiobutton(classifier_frame, text="Decision Tree", variable=self.classifier_var, value="decision_tree", bootstyle="info").grid(row=0, column=2, padx=4)

        type_frame = ttk.Frame(ml_frame)
        type_frame.grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(type_frame, text="Data Type:", width=12, foreground="white").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(type_frame, text="Binary", variable=self.data_type_var, value="binary", bootstyle="info").grid(row=0, column=1, padx=4)
        ttk.Radiobutton(type_frame, text="Multiclass", variable=self.data_type_var, value="multiclass", bootstyle="info").grid(row=0, column=2, padx=4)

        ttk.Button(ml_frame, text="Run ML Pipeline", command=lambda: threading.Thread(target=self.on_ml_button_click).start(), bootstyle="primary").grid(row=2, column=0, pady=6, sticky="w")

        # ---------------------------
        # ML report
        # ---------------------------
        report_frame = ttk.Labelframe(main_frame, text=" ML Report ", bootstyle="secondary")
        report_frame.grid(row=8, column=0, sticky="nsew", padx=6, pady=6)
        report_frame.rowconfigure(0, weight=1)
        report_frame.columnconfigure(0, weight=1)
        mono_font = tkFont.Font(family="Courier New", size=10)
        self.report_text = tk.Text(report_frame, height=12, bg="#0f1112", fg="white",
                                   insertbackground="white", font=mono_font)
        self.report_text.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # ---------------------------
        # Bottom frame
        # ---------------------------
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=9, column=0, sticky="ew", pady=12)
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)

        self.logo1_img = Image.open(r"logos\TEAMHub.jpg")
        self.logo1_img = self.logo1_img.resize((120, 60), Image.Resampling.LANCZOS)
        self.logo1_photo = ImageTk.PhotoImage(self.logo1_img)

        self.logo2_img = Image.open(r"logos\TEBLab.jpg")
        self.logo2_img = self.logo2_img.resize((120, 60), Image.Resampling.LANCZOS)
        self.logo2_photo = ImageTk.PhotoImage(self.logo2_img)

        ttk.Label(bottom_frame, image=self.logo1_photo).grid(row=0, column=0, sticky="w", padx=12)
        ttk.Label(bottom_frame, image=self.logo2_photo).grid(row=0, column=1, sticky="e", padx=12)

        buttons_frame = ttk.Frame(bottom_frame)
        buttons_frame.grid(row=1, column=0, columnspan=2, pady=6)
        ttk.Button(buttons_frame, text="Start", command=self.run_analysis_thread, bootstyle="success").grid(row=0, column=0, padx=12)
        ttk.Button(buttons_frame, text="Exit", command=root.destroy, bootstyle="danger").grid(row=0, column=1, padx=12)
    # =============================
    # Directory selection
    # =============================
    def browse_input(self):
        """Select folder containing microscopy images."""
        path = filedialog.askdirectory()
        self.input_dir.set(path)
    def browse_output(self):
        """Select folder where results will be saved."""
        path = filedialog.askdirectory()
        self.output_dir.set(path)

    # Log
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    # Selected techniques
    def get_selected_techniques(self):
        return [k for k, v in self.techniques.items() if v.get()]

    # --- Radiomics ---
    def run_analysis(self):
        """
        Runs radiomics texture analysis on all images in the input folder.

        Biological context:
        - Input images may be microscopy images of cells, tissues, or organoids. Analysis has solely been focused on mitochondria
        - Extracts quantitative features (intensity, texture) to characterize mitochondrial structures.
        - Saves features for downstream ML classification (e.g., SVM, decision tree).

        Steps:
        1. Validate input/output folders.
        2. Find all TIFF files (common microscopy format).
        3. Initialize radiomics feature extractor with selected feature classes.
        4. Process each image: threshold to identify foreground, extract features.
        5. Save aggregated CSV and Excel files (all features + separated by feature class).
        6. Update GUI progress bar and logging.
        """
        input_path = self.input_dir.get()
        output_path = self.output_dir.get()
        if not os.path.isdir(input_path) or not os.path.isdir(output_path):
            messagebox.showerror("Error", "Invalid input/output directory")
            return

        tiff_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.tif', '.tiff'))]
        if not tiff_files:
            self.log("No TIFF files found.")
            return

        self.log(f"Found {len(tiff_files)} TIFF files. Starting analysis...")
        radiomics.setVerbosity(logging.INFO)
        global logger
        logger = radiomics.logger
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(filename='radiomics_log.txt', mode='w')
        formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        settings = {'binWidth':25, 'resampledPixelSpacing':None, 'interpolator':sitk.sitkBSpline}
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.disableAllFeatures()
        for fclass in self.get_selected_techniques():
            extractor.enableFeatureClassByName(fclass)

        self.progress['maximum'] = len(tiff_files)
        self.progress['value'] = 0
        result_data = []

        for i, imageFile in enumerate(tiff_files):
            imagePath = os.path.join(input_path, imageFile)
            self.log(f"Processing: {imageFile}")
            data = process_image_file(imagePath, extractor)
            if data:
                result_data.append(data)
            self.progress_queue.put((i+1, len(tiff_files)))

        if result_data:
            df = pd.concat([pd.DataFrame(d) for d in result_data], ignore_index=True)
            df.columns = [rename_columns(c) for c in df.columns]
            all_csv_path = os.path.join(output_path, 'radiomics_features.csv')
            df.to_csv(all_csv_path, index=False)
            all_excel_path = os.path.join(output_path, 'radiomics_features.xlsx')
            sep_excel_path = os.path.join(output_path, 'radiomics_features_separated.xlsx')
            with pd.ExcelWriter(all_excel_path, engine='xlsxwriter') as writer:
                df.drop(columns=['Image Name']).to_excel(writer, sheet_name='All Features', index=False)
            with pd.ExcelWriter(sep_excel_path, engine='xlsxwriter') as writer:
                for feature in self.get_selected_techniques():
                    cols = [c for c in df.columns if c.startswith(feature)] + ['Image Name']
                    df[cols].to_excel(writer, sheet_name=f'{feature.upper()} Features', index=False)
            self.log(f"Analysis complete. Results saved to:\n{all_csv_path},\n{all_excel_path}, and\n{sep_excel_path}")
        else:
            self.log("No data to write.")

    def run_analysis_thread(self):
        threading.Thread(target=self.run_analysis, daemon=True).start()

    # --- RFE ---
    def handle_rfe_filter(self):
        """
        Runs Recursive Feature Elimination (RFE) to select informative features.

        Biological context:
        - Many radiomic features may be correlated or redundant.
        - RFE ranks features by importance using a Random Forest classifier.
        - Top-ranked features correspond to most discriminative biological descriptors.
        - Reduces feature dimensionality, improving ML interpretability.
        """
        input_file = filedialog.askopenfilename(filetypes=[("CSV Files","*.csv")])
        if input_file:
            output_dir = filedialog.askdirectory(title="Select Output Directory")
            if output_dir:
                self.log("Running RFE feature selection...")

                def run_and_log():
                    try:
                        self.progress.config(mode='indeterminate')
                        self.progress.start(10)

                        summary, reduced_path, summary_path = run_rfe_feature_selection(
                            input_file,
                            output_dir,
                            top_k=20,
                            n_estimators=50,
                            n_jobs=1
                        )

                        self.progress.stop()
                        self.progress.config(mode='determinate', value=100)
                        self.progress_label.config(text="Progress: 100%")

                        self.log(f"RFE complete. Reduced dataset: {reduced_path}")
                        if summary_path:
                            self.log(f"Summary saved to: {summary_path}")
                        self.log(summary.head(10).to_string(index=False))

                    except Exception as e:
                        self.progress.stop()
                        self.progress.config(mode='determinate', value=0)
                        self.progress_label.config(text="Progress: 0%")
                        self.log(f"Error during RFE: {e}")

                threading.Thread(target=run_and_log, daemon=True).start()

    # --- ML ---
    def on_ml_button_click(self):
        """
        Executes ML classification (Decision Tree or SVM) on pre-extracted biological features.

        Biological context:
        - Input CSV contains features extracted from images.
        - Binary or multiclass classification corresponds to experimental groups such as mitochondrial structure type or applied treatments.
        - Output includes test set accuracy, cross-validation scores, confusion matrix, and ROC curves.
        """
        try:
            classifier = self.classifier_var.get()
            data_type = self.data_type_var.get()
            self.log(f"Running {data_type.upper()} ML pipeline with {classifier.upper()}...")
            input_csv = filedialog.askopenfilename(filetypes=[("CSV Files","*.csv")])
            if not input_csv:
                self.log("ML pipeline cancelled: no input file selected.")
                return

            # Call the appropriate function
            if classifier == "decision_tree":
                if data_type == "binary":
                    report = decisiontree_binary(input_csv)
                else:
                    report = decisiontree_multiclass(input_csv)
            else:  # SVM
                if data_type == "binary":
                    report = svm_binary(input_csv)
                else:
                    report = svm_multiclass(input_csv)

            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(tk.END, report)

        except Exception as e:
            self.log(f"Error during ML pipeline: {e}")

    # --- Progress update ---
    def update_progress_from_queue(self):
        try:
            while not self.progress_queue.empty():
                item = self.progress_queue.get_nowait()
                if isinstance(item, tuple):
                    current, total = item
                    self.progress['value'] = current
                    percent = int((current/total)*100)
                    self.progress_label.config(text=f"Progress: {percent}%")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.update_progress_from_queue)


# Launch GUI
if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    app = RadiomicsApp(root)
    root.mainloop()
