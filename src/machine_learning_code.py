import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline

# =============================
# Decision Tree
# =============================
def decisiontree(input_file, root=None, problem_type="binary", text_widget=None):
    """
    Run a decision tree classifier on features extracted relating to the mitochondrial data

    Parameters
    ----------

    input_file : str
        CSV file containing the dataset; features extracted from the PyRadiomics 
        library, last column is assumed to be the class label
    root : Tkinter root, optional
        If the root is provided, the confusion matrix plots will be embedded in the GUI
    problem_type : str
        'binary' or 'multiclass'; defines the type of classification
    text_widget : Tkinter Text widget, optional
        If this is provided, outputs will be printed in the GUI text box

    Return
    ----------

    output_text : str
        Summary of the classification results including cross-validation accuracy, 
        text accuracy, and classification report for each class
    """
    # Load dataset; drop samples with missing class labels
    data = pd.read_csv(input_file)
    data = data.dropna(subset=[data.columns[-1]])

    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    #I dentify all unique experimental conditions such as structure types and treatments
    classes = np.unique(y)
    datasetLabels = [str(c) for c in classes]

    # Stratified split: train/test ensures each condition is represented proportionally
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1, stratify=y
    )

    # Normalize feature measurements for fair comparison
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Decision Tree and hyperparameter tuning:
    # Finds the tree depth and splitting criteria that best classify the experimental condition
    model = DecisionTreeClassifier(random_state=1)
    param_grid = {
        'criterion': ['gini', 'entropy'], # splitting based on class purity
        'max_depth': [None, 10, 20, 30, 40, 50], # prevents overfitting small experimental datasets
        'min_samples_split': [2, 5, 10], # controls how small a subgroup must be in order to split
        'min_samples_leaf': [1, 2, 4] # ensures each terminal  node has enough samples
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    # Evaluate robustness across multiple splits of the entire dataset
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=1)
    cv_scores = cross_val_score(best_model, scaler.fit_transform(X), y, cv=cv, n_jobs=-1)
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    # Train best model and classify the test samples
    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)

    # Confusion matrix plot: shows how many sample from each biological 
    # condition (i.e., mitochondrial structure or treatment) were correctly classifier
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=datasetLabels)
    if root:
        fig, ax = plt.subplots(figsize=(5,4))
        disp.plot(ax=ax, cmap="Blues", values_format='d')
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=11, column=0, columnspan=4)
    else:
        disp.plot(cmap="Blues", values_format='d')
        plt.show()

    # ===== ROC Curve =====
    y_score = best_model.predict_proba(X_test_scaled)
    if len(classes) == 2:  # Binary ROC: e.g., control vs treated (LPSIFN)
        prob_pos = y_score[:, 1] if y_score.shape[1] == 2 else y_score[:, 0]
        fpr, tpr, _ = roc_curve(y_test, prob_pos, pos_label=classes[1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Binary ROC Curve (Decision Tree)')
        plt.legend(loc='lower right')
        plt.show()
    else:  # Multiclass ROC: compares performance across multiple classes (i.e., mitochondrial structures)
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = y_score.shape[1]
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        roc_auc["macro"] = auc(all_fpr, mean_tpr)

        plt.figure()
        colors = plt.cm.get_cmap('tab10', n_classes)
        for i, label in enumerate(datasetLabels):
            plt.plot(fpr[i], tpr[i], color=colors(i), lw=2, label=f'{label} (AUC={roc_auc[i]:.2f})')
        plt.plot([0,1],[0,1],'k--', lw=2)
        plt.plot(all_fpr, mean_tpr, color='navy', lw=2, linestyle='--', label=f'Macro AUC={roc_auc["macro"]:.2f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass ROC Curve (Decision Tree)')
        plt.legend(loc='lower right')
        plt.show()

    # Classification report
    output_text = (
        f"{'─'*100}\n"
        f"Decision Tree Classification\n"
        f"{'─'*100}\n"
        f"CV Accuracy: {cv_mean:.2f} ± {cv_std:.2f}\n"
        f"Test Accuracy: {test_acc:.2f}\n"
        f"{'─'*100}\n"
        f"{classification_report(y_test, y_pred, target_names=datasetLabels)}\n"
        f"{'─'*100}"
    )
    if text_widget:
        text_widget.insert('end', output_text + '\n')

    return output_text


# =============================
# SVM
# =============================
def supervised_vector_machine(input_file, root=None, problem_type="multiclass", text_widget=None):
    """
    Run a support vector machine (SVM) classifier on features extracted relating to the mitochondrial data

    Parameters
    ----------

    input_file : str
        CSV file containing the dataset; features extracted from the PyRadiomics 
        library, last column is assumed to be the class label
    root : Tkinter root, optional
        If the root is provided, the confusion matrix plots will be embedded in the GUI
    problem_type : str
        'binary' or 'multiclass'; defines the type of classification
    text_widget : Tkinter Text widget, optional
        If this is provided, outputs will be printed in the GUI text box

    Return
    ----------

    output_text : str
        Summary of the classification results including cross-validation accuracy, 
        text accuracy, and classification report for each class
    """

    # Load dataset and drop missing values
    data = pd.read_csv(input_file)
    data = data.dropna()
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    classes = np.unique(y)
    datasetLabels = [str(c) for c in classes]

    # Train-test split
    # Stratified split preserves biological label distribution in training/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1, stratify=y
    )

    # Scaling + SVM
    # Standardization ensure features are comparable
    classifier = SVC(kernel='rbf', probability=True, random_state=1)
    scaler = StandardScaler()
    pipeline = make_pipeline(scaler, classifier)

    # Cross-validation
    # Repeated stratified k-fold evaluates stability of classifier on the biological data
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    cv_mean = np.mean(scores)
    cv_std = np.std(scores)

    # Train on training set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=datasetLabels)
    if root:
        fig, ax = plt.subplots(figsize=(5,4))
        disp.plot(ax=ax, cmap="Blues", values_format='d')
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=11, column=0, columnspan=4)
    else:
        disp.plot(cmap="Blues", values_format='d')
        plt.show()

    # ===== ROC Curve =====
    # Visualizes the classifier discrimination between the different classes
    y_score = pipeline.predict_proba(X_test)
    if len(classes) == 2:  # Binary case: Control vs LPSIFN treated BMDMs
        prob_pos = y_score[:, 1] if y_score.shape[1] == 2 else y_score[:, 0]
        fpr, tpr, _ = roc_curve(y_test, prob_pos, pos_label=classes[1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Binary ROC Curve (SVM)')
        plt.legend(loc='lower right')
        plt.show()
    else:  # Multiclass: Comparing fibers, puncta, and rods
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = y_score.shape[1]
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        roc_auc["macro"] = auc(all_fpr, mean_tpr)

        plt.figure()
        colors = plt.cm.get_cmap('tab10', n_classes)
        for i, label in enumerate(datasetLabels):
            plt.plot(fpr[i], tpr[i], color=colors(i), lw=2, label=f'{label} (AUC={roc_auc[i]:.2f})')
        plt.plot([0,1],[0,1],'k--', lw=2)
        plt.plot(all_fpr, mean_tpr, color='navy', lw=2, linestyle='--', label=f'Macro AUC={roc_auc["macro"]:.2f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass ROC Curve (SVM)')
        plt.legend(loc='lower right')
        plt.show()

    # Classification report
    output_text = (
        f"{'─'*100}\n"
        f"SVM Classification\n"
        f"{'─'*100}\n"
        f"Best CV Accuracy: {cv_mean:.2f} ± {cv_std:.2f}\n"
        f"Test Accuracy: {test_acc:.2f}\n"
        f"{'─'*100}\n"
        f"{classification_report(y_test, y_pred, target_names=datasetLabels)}\n"
        f"{'─'*100}"
    )
    if text_widget:
        text_widget.insert('end', output_text + '\n')

    return output_text


# =============================
# Wrapper functions for GUI
# =============================
def decisiontree_binary(*args, **kwargs):
    return decisiontree(*args, problem_type="binary", **kwargs)

def decisiontree_multiclass(*args, **kwargs):
    return decisiontree(*args, problem_type="multiclass", **kwargs)

def svm_binary(*args, **kwargs):
    return supervised_vector_machine(*args, problem_type="binary", **kwargs)

def svm_multiclass(*args, **kwargs):
    return supervised_vector_machine(*args, problem_type="multiclass", **kwargs)

