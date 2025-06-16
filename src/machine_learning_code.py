import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer  


# Decision Tree model function
def decisiontree(input_file):
    # Load the dataset
    data = pd.read_csv(input_file)
    data = data.dropna(subset=[data.columns[-1]])

    # Mapping and labels for target column
    unique_classes = np.unique(data.iloc[:, -1])
    datasetMapping = {k: str(k) for k in unique_classes}
    datasetLabels = [str(k) for k in unique_classes]
    classes = list(unique_classes)

    # Features and target variable
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Binarize the output labels for ROC curve (required for multi-class problems)
    y_bin = label_binarize(y, classes=classes)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # Initialize StandardScaler for feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the model and hyperparameters for Decision Tree
    model = DecisionTreeClassifier(random_state=1)
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform grid search with cross-validation for Decision Tree
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    # Best model for Decision Tree
    best_model = grid_search.best_estimator_

    # Perform cross-validation on the entire dataset for Decision Tree
    cv = StratifiedKFold(n_splits=10)
    cv_scores = cross_val_score(best_model, scaler.fit_transform(X), y, cv=cv, scoring='accuracy')

    # Print cross-validation scores for Decision Tree
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {np.mean(cv_scores):.2f}")
    print(f"Standard deviation of cross-validation scores: {np.std(cv_scores):.2f}")

    # Fit the best Decision Tree model on the training data
    best_model.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)  # Get class probabilities for ROC curve

    # Print confusion matrix for Decision Tree
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=datasetLabels)
    disp.plot()
    plt.show()

    # Print classification report for Decision Tree
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=datasetLabels))

    # Compute ROC-AUC for Decision Tree
    y_test_bin = label_binarize(y_test, classes=classes)
    if y_test_bin.shape[1] == 1:
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

    y_score = best_model.predict_proba(X_test_scaled)

    fpr, tpr, _ = roc_curve(y_test_bin[:, 1], y_score[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for Decision Tree
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def supervised_vector_machine(input_file):
    # Load the dataset
    data = pd.read_csv(input_file)
    data = data.dropna()

    # Extract features and labels
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values

    # Dynamically generate label mapping
    unique_classes = sorted(np.unique(Y))
    datsetMapping = {val: f'Class{val}' for val in unique_classes}
    datasetLabels = list(datsetMapping.values())
    classes = unique_classes

    # Dataset info
    numberFeatures = len(X[0])
    numberCat = len(classes)

    # Scale data
    sc = StandardScaler()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=1)

    # Define model pipeline
    classifier = SVC(kernel='rbf', probability=True)
    LDA = LinearDiscriminantAnalysis()
    modelPipeline = make_pipeline(sc, LDA, classifier)

    # Cross-validation
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=1)
    scores = cross_val_score(modelPipeline, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)

    accuracyKScore = np.mean(scores)
    stdKScore = np.std(scores)

    # Train and predict
    modelPipeline.fit(X_train, Y_train)
    Y_pred = modelPipeline.predict(X_test)
    accuracyScore = accuracy_score(Y_test, Y_pred)

    # Map numerical to categorical labels
    try:
        Y_pred = [datsetMapping[i] for i in Y_pred]
        Y_test = [datsetMapping[i] for i in Y_test]
    except KeyError as e:
        print("Error during ML pipeline:", e)
        print(f"Label {e} not found in dataset mapping: {datsetMapping}")
        return

    # Print results
    print()
    print('─' * 100)
    print("Data Set: RFE")
    print("Kernel: RBF")
    print('─' * 100)
    print("Best Accuracy: %.2f +/- %.2f" % (accuracyKScore, stdKScore))
    print('─' * 100)
    print(classification_report(Y_test, Y_pred))
    print('─' * 100)

    # Confusion matrix
    cm = confusion_matrix(Y_test, Y_pred, labels=datasetLabels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=datasetLabels)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # ROC curve (binary classification only)
    if len(classes) == 2:
        probs = modelPipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve([1 if y == datasetLabels[1] else 0 for y in Y_test], probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()