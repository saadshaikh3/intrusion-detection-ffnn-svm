# Intrusion Attack Detection Using Neural Networks and Support Vector Machine  

## Overview  
This project utilizes the **KDD Cup 1999 dataset subset** to develop a classification model for network intrusion detection. The objective is to classify network connection records as:  
- **Normal (0)**  
- **Abnormal/Attack (1)**  

Two models were compared for their classification performance:  
- **Feedforward Neural Network (ANN)**  
- **Support Vector Machine (SVM)**  

---

## Dataset  
The dataset contains:  
- **494,021 rows** and **41 input features** with 1 target column.  
- Target classes were simplified into binary:  
  - **Normal**: `0`  
  - **Attack**: `1`  

---

## Methodology  
### 1. Data Preparation  
- Categorical features were encoded into numerical format.  
- Target classes (23) were simplified into binary.  
- Columns with zero variance (columns 20 and 21) were removed to reduce dimensions.  

### 2. Data Preprocessing  
- Input features were normalized using **min-max scaling** to bring all values into the range `[0,1]`.  
- This ensures better generalization during training.  

### 3. Model Training  
- **Feedforward Neural Network (ANN)**:  
  - Tested various topologies with different hidden layer sizes and learning rates.  
  - Cross-validation: **5 folds**.  
  - Training function: **trainlm**.  

- **Support Vector Machine (SVM)**:  
  - Trained using a **linear kernel**.  
  - Same cross-validation approach applied.  

### 4. Evaluation Metrics  
- **Mean Squared Error (MSE)**  
- **Accuracy**  
- **F1-Score**  

---

## Results Summary  
The **Feedforward Neural Network (ANN)** outperformed the SVM for this dataset:  
- **F1-Score**: 99.946%  
- **Accuracy**: 99.913%  
- **MSE**: Lowest achieved with:  
  - Hidden Layer Size: **30**  
  - Learning Rate: **0.05**  

---

## Future Directions  
- Experimenting with additional models such as **Random Forest**.  
- Increasing the number of hidden layers or neurons in ANN.  
- Fine-tuning hyperparameters and extending the training epochs.  

---

## Files in Repository  
- **Notebook**: `intrusion_detection.ipynb`  
- **Dataset**: KDD Cup 1999 subset csv file  


---

## Acknowledgements  
- **KDD Cup 1999 Dataset**  
- **Neural Network and SVM Frameworks**  
