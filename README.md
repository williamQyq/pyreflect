# **NR-SCFT-ML: Neutron Reflectivity SCFT Machine Learning**
A Python package for **Chi parameters prediction** from **SLD profile data** and **SLD prediction from NR curves** using machine learning.

## **Overview**
NR-SCFT-ML is a PyPI package designed to preprocess Neutron Reflectivity (NR) data, train an Autoencoder + MLP model, and predict Chi parameters. The tool simplifies data preparation, training, and inference with just two commands. A quick CLI is provided for fast neutron reflectivity analysis.
## **Quick Start Guide**

### 🧑🏻‍💻Example Notebooks ###
To learn how to use it, check the example notebooks in `pyreflect/example_notebooks`  

### **1️⃣ Initialize Configuration**
To set up the default configuration, run:

```bash
python -m pyreflect init
```  
### **2️⃣ Run Training & Prediction**
Open `settings.yml` and update the file paths for your SLD profile data and Chi parameters

### **3️⃣ Run interaction chi parameters prediction**
```bash
python -m pyreflect run --enable-chi-prediction
```

### **4️⃣ Run sld profile prediction from nr curves**
```bash
python -m pyreflect run --enable-sld-prediction
```