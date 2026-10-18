# ECG Arrhythmia Classification Using Machine Learning

## 1. Project Overview

This project investigates ECG heartbeat classification using the MIT-BIH Arrhythmia Database. It processes ECG recordings, segments individual heartbeats, maps beat annotations to the five AAMI classes, extracts features, splits retained beats, and trains a Random Forest classifier using SMOTE and to address class imbalance.

The project also evaluates beat-level classification performance, examines patient-level predictions.

## 2. Dataset

The project uses the MIT-BIH Arrhythmia Database, available through PhysioNet:

https://physionet.org/content/mitdb/

The dataset is downloaded locally by the data-provenance script. The downloaded recordings are stored in a folder named `mitdb/`.

Please follow the dataset's access conditions and citation requirements.

## 3. Desktop Setup

Clone or download this repository to a folder in your computer, and open a terminal in the project directory.
Download VS Code on your desktop.
Open VS Code.
Go to  Files on the top left panel, then go to Open Folder, then go to the Folder you cloned the repository to in your computer, then click open.
Open the terminal, this will open the directorry in which the repository is on.

## 4. Requirements

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- WFDB
- NeuroKit2
- Scikit-learn
- Imbalanced-learn
- Joblib

## 5. Installation

Before creating a virtual environment, run the following command in the current terminal session:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Create and activate a virtual environment before the code execution, then install the dependencies in the requirements.

```bash
py -m venv .venv

.venv\Scripts\Activate.ps1

python -c "import sys; print(sys.executable)"
```

Install the required packages in the requirements text file using the below command:

pip install numpy pandas scipy scikit-learn imbalanced-learn wfdb neurokit2 matplotlib

```bash
python -m pip install -r requirements.txt
```

## 6. Execution Order

Run the main pipeline scripts from the project directory in the following order:

1. `step1_data_provenance.py` - downloads and examines the ECG dataset.
2. `step2_preprocessing.py` - contains the ECG signal preprocessing functions used by later stages.
3. `step3_segmentation.py` - segments heartbeats and generates the AAMI labels and patient identifiers.
4. `step4_patient_labeling.py` - generates patient-level labels.
5. `step5_feature_extraction_relaxed.py` - extracts heartbeat features while retaining selected missing values.
6. `step7_train_test_split_relaxed.py` - creates the training and testing datasets and handles missing features.
7. `step8_smote_random_forest_gridsearch.py` - tunes and trains the Random Forest pipeline using SMOTE.
8. `step8_evaluation_gridsearch.py` - evaluates beat-level and patient-level predictions.

Run each script only after the preceding stage has completed successfully.

For example:
Note: the following are the prompts you must call on your terminal for each step respectively.
```bash
python step1_data_provenance.py
python step2_preprocessing.py
python step3_segmentation.py
python step4_patient_labeling.py
python step5_feature_extraction_relaxed.py
python step7_train_test_split_relaxed.py
python step8_smote_random_forest_gridsearch.py
python step8_evaluation_gridsearch.py
```

The modelling stage also requires the `smote_utils.py` helper module to be present in the project directory.

## 7. Outputs

The scripts generate intermediate NumPy arrays and CSV files, a trained model, and evaluation results.

These outputs are used by subsequent stages of the pipeline. They are generated locally when the scripts execute.

## 8. Reproducibility

Run the scripts in the stated order and use the same Python environment throughout the pipeline. Ensure that the dataset is downloaded and all dependencies are installed before executing the processing and modelling stages.
