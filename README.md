# **NR-SCFT-ML: Neutron Reflectivity SCFT Machine Learning**
A Python package for **Chi parameters prediction** from **SLD profile data** and **SLD prediction from NR curves** using machine learning.

## **Overview**
**NR-SCFT-ML** is a PyPI package for efficient processing and analysis of Neutron Reflectivity (NR) data. It provides a streamlined pipeline for:

- Preprocessing NR datasets  
- Predicting SLD Profile by training a CNN
- Predicting **Chi** parameters by training a combined Autoencoder and MLP model.

## **Quick Start Guide**
To install from TestPyPI, run:
```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pyreflect==1.1.7
```
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

## Credits

This project builds on work by:
- Prof. Miguel Fuentes-Cabrera [NR-SCFT-ML](https://github.com/miguel-fc/NR-SCFT-ML)
- Dr. Rajeev Kumar
- Brian Qu ([NR-SLD-CNN](https://github.com/BBQ591/NR-SLD-CNN))

## Author

- Yuqing Qiao (William) – Maintainer and developer of this PyPI package
