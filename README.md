# **NR-SCFT-ML: Neutron Reflectivity SCFT Machine Learning**
A Python package for **Chi parameters prediction** from **SLD profile data** using machine learning.

## **Overview**
NR-SCFT-ML is a CLI-based tool designed to preprocess **Neutron Reflectivity (NR) data**, train an **Autoencoder + MLP model**, and predict **Chi parameters**. The tool simplifies the process of **data preparation, training, and inference** with just two commands.

## **Quick Start Guide**

### **1️⃣ Initialize Configuration**
To set up the default configuration, run:

```bash
python -m pyreflect init
```  
### **2️⃣ Run Training & Prediction**
Open `settings.yml` and update the file paths for your SLD profile data and Chi parameters

### **3️⃣ Run Training & Prediction**
```bash
python -m pyreflect run
```