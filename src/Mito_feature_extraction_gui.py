# Import necessary libraries for GUI, file handling, radiomics extraction, and machine learning
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
from PIL import Image
from anova_feature_selection import run_anova_filter  # Custom ANOVA filter script
from machine_learning_code import decisiontree, supervised_vector_machine  # Custom ML script

# Helper function to remove 'original_' prefix from column names
def rename_columns(col):
    return col.replace('original_', '') if col.startswith('original_') else col

# Create a binary mask using Otsu thresholding
def create_thresholded_mask(image_np):
    # Normalize to 8-bit if 16-bit image
    if image_np.dtype == np.uint16:
        image_8bit = (image_np / 256).astype(np.uint8)
    elif image_np.dtype == np.uint8:
        image_8bit = image_np
    else:
        raise ValueError("Unsupported image bit depth. Only 8-bit and 16-bit images are supported.")

    # If RGB or multi-channel, convert to grayscale (use first channel)
    if image_8bit.ndim == 3:
        image_8bit = image_8bit[:, :, 0]

    # Apply Otsu's thresholding to generate a binary mask
    _, mask = cv2.threshold(image_8bit, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask.astype(np.uint8)


def process_image_file(image_path, extractor):
    try:
        image = Image.open(image_path)
        image_np = np.array(image)

        # If RGB/multi-channel, convert to grayscale for consistency
        if image_np.ndim == 3:
            image_np = image_np[:, :, 0]

        # Create Otsu-thresholded binary mask
        mask_np = create_thresholded_mask(image_np)

        # Convert to SimpleITK images
        image_sitk = sitk.GetImageFromArray(image_np)
        mask_sitk = sitk.GetImageFromArray(mask_np)
        mask_sitk.CopyInformation(image_sitk)

        # Extract radiomics features
        featureVector = extractor.execute(image_sitk, mask_sitk)

        # Adjust Sum Average for GLCM
        if 'original_glcm_SumAverage' in featureVector and 'original_glcm_JointAverage' in featureVector:
            featureVector['original_glcm_SumAverage'] = 2 * featureVector['original_glcm_JointAverage']

        # Prepare feature dictionary
        data = {'Image Name': [os.path.basename(image_path)]}
        for feature_class in ['firstorder', 'glcm', 'glrlm', 'glszm', 'ngtdm', 'gldm']:
            for key, value in featureVector.items():
                if key.startswith(f'original_{feature_class}_'):
                    data[key] = [value]

        return data

    except Exception as e:
        logging.error(f"Error processing image '{image_path}': {str(e)}")
        return None


# Main GUI application class
class RadiomicsApp:
    def __init__(self, root):
        self.root = root
        root.title("Radiomics Texture Analysis Tool")

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.technique_var = tk.StringVar()

        # Input and output folder selection
        tk.Label(root, text="Input Folder").grid(row=0, column=0)
        tk.Entry(root, textvariable=self.input_dir, width=50).grid(row=0, column=1)
        tk.Button(root, text="Browse", command=self.browse_input).grid(row=0, column=2)

        tk.Label(root, text="Output Folder").grid(row=1, column=0)
        tk.Entry(root, textvariable=self.output_dir, width=50).grid(row=1, column=1)
        tk.Button(root, text="Browse", command=self.browse_output).grid(row=1, column=2)

        # Texture feature selection checkboxes
        tk.Label(root, text="Texture Techniques").grid(row=2, column=0, sticky='nw')
        self.techniques = {
            'firstorder': tk.BooleanVar(value=True),
            'glcm': tk.BooleanVar(value=True),
            'glrlm': tk.BooleanVar(value=True),
            'glszm': tk.BooleanVar(value=True),
            'ngtdm': tk.BooleanVar(value=True),
            'gldm': tk.BooleanVar(value=True)
        }

        checkbox_frame = tk.Frame(root)
        checkbox_frame.grid(row=2, column=1, sticky='w')

        for idx, (technique, var) in enumerate(self.techniques.items()):
            tk.Checkbutton(checkbox_frame, text=technique.upper(), variable=var).grid(row=idx // 3, column=idx % 3, sticky='w')

        # Buttons for running analysis and ANOVA filtering
        tk.Button(root, text="Run Analysis", command=self.run_analysis_thread).grid(row=3, column=1, pady=10)
        tk.Button(root, text="Run ANOVA Filter", command=self.handle_anova_filter).grid(row=3, column=2, pady=10)
        # Progress bar and label
        self.progress = ttk.Progressbar(root, orient='horizontal', mode='determinate', length=400)
        self.progress.grid(row=4, column=0, columnspan=3, pady=(0, 10))
        self.progress_label = tk.Label(root, text="Progress: 0%")
        self.progress_label.grid(row=4, column=3, padx=(10, 0))
        self.progress_queue = queue.Queue()
        self.root.after(100, self.update_progress_from_queue)  # Periodically update progress bar

        # Scrolled text area for logging output
        self.log_text = scrolledtext.ScrolledText(root, width=80, height=20)
        self.log_text.grid(row=5, column=0, columnspan=3)
        
        # Machine learning classifier selection
        self.classifier_var = tk.StringVar(value="decision_tree")
        tk.Label(root, text="Select Classifier:").grid(row=6, column=0)
        tk.Radiobutton(root, text="SVM", variable=self.classifier_var, value="svm").grid(row=6, column=1)
        tk.Radiobutton(root, text="Decision Tree", variable=self.classifier_var, value="decision_tree").grid(row=6, column=2)

        tk.Button(root, text="Run ML Pipeline", command=lambda: threading.Thread(target=self.on_ml_button_click).start()).grid(row=9, column=0, columnspan=3, pady=10)

        # Report output from ML pipeline
        self.report_text = tk.Text(root, height=15, width=100)
        self.report_text.grid(row=10, column=0, columnspan=3)

    # Browse input/output directories
    def browse_input(self):
        path = filedialog.askdirectory()
        self.input_dir.set(path)

    def browse_output(self):
        path = filedialog.askdirectory()
        self.output_dir.set(path)

    # Log messages to the GUI
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    # Return list of selected feature classes
    def get_selected_techniques(self):
        return [k for k, v in self.techniques.items() if v.get()]
    

    # Main function to run radiomics feature extraction
    def run_analysis(self):
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

        # Set up radiomics logging
        radiomics.setVerbosity(logging.INFO)
        global logger
        logger = radiomics.logger
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(filename='radiomics_log.txt', mode='w')
        formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Set extractor parameters
        settings = {'binWidth': 25, 'resampledPixelSpacing': None, 'interpolator': sitk.sitkBSpline}
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.disableAllFeatures()
        for fclass in self.get_selected_techniques():
            extractor.enableFeatureClassByName(fclass)

        # Initialize progress
        self.progress['maximum'] = len(tiff_files)
        self.progress['value'] = 0

        result_data = []

        # Process each image
        for i, imageFile in enumerate(tiff_files):
            imagePath = os.path.join(input_path, imageFile)
            self.log(f"Processing: {imageFile}")
            data = process_image_file(imagePath, extractor)
            if data:
                result_data.append(data)

            self.progress_queue.put((i + 1, len(tiff_files)))

        # Save results
        if result_data:
            df = pd.concat([pd.DataFrame(d) for d in result_data], ignore_index=True)
            df.columns = [rename_columns(c) for c in df.columns]

            #Saves results as both a csv and excel and separated excel sheets to avoid confusion as well
            all_csv_path = os.path.join(output_path, 'radiomics_features.csv')
            df.to_csv(all_csv_path, index=False)

            all_excel_path = os.path.join(output_path, 'radiomics_features.xlsx')
            sep_excel_path = os.path.join(output_path, 'radiomics_features_separated.xlsx')

            # Save full and separated Excel files
            with pd.ExcelWriter(all_excel_path, engine='xlsxwriter') as writer:
                df.drop(columns=['Image Name']).to_excel(writer, sheet_name='All Features', index=False)

            with pd.ExcelWriter(sep_excel_path, engine='xlsxwriter') as writer:
                for feature in self.get_selected_techniques():
                    cols = [c for c in df.columns if c.startswith(feature)] + ['Image Name']
                    df[cols].to_excel(writer, sheet_name=f'{feature.upper()} Features', index=False)

            self.log(f"Analysis complete. Results saved to:\n{all_csv_path},\n{all_excel_path}, and\n{sep_excel_path}")
        else:
            self.log("No data to write.")

    def update_progress_from_queue(self):
        try:
            while not self.progress_queue.empty():
                current, total = self.progress_queue.get_nowait()
                self.progress['value'] = current
                percent = int((current / total) * 100)
                self.progress_label.config(text=f"Progress: {percent}%")
        except queue.Empty:
            pass
        finally:
            # Check again after 100ms
            self.root.after(100, self.update_progress_from_queue)


    # Start analysis in a new thread to avoid freezing GUI
    def run_analysis_thread(self):
        analysis_thread = threading.Thread(target=self.run_analysis)
        analysis_thread.start()

    # Handle running the ANOVA feature filter
    def handle_anova_filter(self):
        input_file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if input_file:
            output_dir = filedialog.askdirectory(title="Select Output Directory")
            if output_dir:
                self.log("Running ANOVA feature selection...")
                run_anova_filter(input_csv=input_file, output_dir=output_dir, top_k=10)  # Run ANOVA with top 10 features

    # Handle ML button click and execute selected classifier  

    def on_ml_button_click(self):
        try:
            classifier = self.classifier_var.get()
            self.log(f"Running machine learning pipeline with {classifier.upper()}...")

            # Prompt for input file (filtered features, e.g., ANOVA output)
            input_csv = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
            if not input_csv:
                self.log("ML pipeline cancelled: no input file selected.")
                return

            # Call the ML pipeline function
            if classifier == "decision_tree":
                report = decisiontree(input_csv)
            else:  # SVM
                report = supervised_vector_machine(input_csv)

        except Exception as e:
            self.log(f"Error during ML pipeline: {e}")

   
# Launch the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = RadiomicsApp(root)
    root.mainloop()
