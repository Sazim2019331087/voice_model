# **Deep Learning-Based Speech Recognition Using Audio Processing**

## **Overview**  
This project aims to build an **Deep Learning Powered Speech Recognition System** capable of identifying individuals based on their voice data. We have collected voice samples through our custom-built website and trained a deep learning model using these samples to achieve speaker recognition.

---

## **Key Features**  
- **Custom Data Collection Platform**:  
  - Website: [sustaudio.zya.me](https://sustaudio.zya.me)  
  - Collected user data: Email, Name, Age, Gender, Institute, Department/Class, Language.  
  - Recorded voice samples (1–3 minutes long).  
  - Total collected: **68 audio samples** from **36 unique individuals**.

- **Deep Learning Model**:  
  - Extracted **MFCC (Mel-Frequency Cepstral Coefficients)** features from audio files.  
  - Built a **Convolutional Neural Network (CNN)** using TensorFlow.  
  - Achieved the following results:  
    - **Accuracy**: 77.78%  
    - **Precision**: 69%  
    - **Recall**: 78%  
    - **F1-Score**: 71%.

- **Results and Analysis**:  
  - Training and validation loss curves.  
  - Confusion matrix with annotated results.  
  - Precision, Recall, and F1-Score visualization as bar charts.  

---

## **Project Workflow**

### **1. Data Collection**  
- Developed a website to collect user information and voice samples.  
- Ensured high-quality audio samples by providing detailed instructions in **English and Bangla**.  
- Stored user data and audio files in a structured format for processing.  

### **2. Dataset Preparation**  
- Created a `dataset.csv` file with the following columns:  
  `email`, `name`, `age`, `gender`, `institute`, `dept_or_class`, `language`, `audio_file`.  
- Encoded `email` as numerical labels using `LabelEncoder`.  

### **3. Feature Extraction**  
- Extracted **MFCC features** using `librosa` with the following parameters:  
  - `n_mfcc = 13` (number of MFCC features).  
  - `max_len = 130` (ensures uniform input size).  

### **4. Model Development**  
- Built a CNN using TensorFlow and Keras:  
  - Input Shape: `(13, 130)` (MFCC features).  
  - Layers: Convolutional, MaxPooling, Flatten, Dense, Dropout.  
  - Optimizer: Adam.  
  - Loss Function: Sparse Categorical Cross-Entropy.  

### **5. Evaluation and Testing**  
- Evaluated model performance on validation data.  
- Tested the model using unseen audio samples for speaker recognition.

---

## **Code Structure**
- **`voice_analysis.ipynb`**: Main notebook containing the implementation, including:  
  - Data processing and feature extraction.  
  - Model development and training.  
  - Performance evaluation and visualization.

- **`dataset.csv`**: Contains the metadata of collected audio samples.  
- **`voices/`**: Directory with `.wav` audio files used for training and testing.  

---

## **Usage**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Sazim2019331087/voice_model.git
   cd voice-model
   ```

2. **Install Required Libraries**:
   ```bash
   pip install librosa numpy pandas scikit-learn tensorflow matplotlib
   ```

3. **Run the Notebook**:
   Open and execute `voice_model.ipynb` in Jupyter Notebook or Google Colab.

4. **Test the Model**:
   Use the `predict_speaker` function to test the model with new audio files. Example:
   ```python
   predicted_email = predict_speaker(model, 'path_to_audio_file.wav')
   print("Predicted Speaker Email:", predicted_email)
   ```

---

## **Results and Visualization**  

### **Performance Metrics**:
| Metric       | Value  |
|--------------|--------|
| Accuracy     | 77.78% |
| Precision    | 69%    |
| Recall       | 78%    |
| F1-Score     | 71%    |

### **Visualization**:
- **Training and Validation Loss Curve**:  
  Showcases model convergence over epochs.  

- **Confusion Matrix**:  
  Displays true and predicted labels, helping analyze misclassifications.  

- **Precision, Recall, and F1-Score Bar Chart**:  
  Visual representation of key performance metrics.

---

## **Credits**  
### **Supervisor**:  
- **Md. Shadmim Hasan Sifat**  
  Lecturer, Dept. of Computer Science and Engineering, SUST.

### **Project Contributor**:  
- **Md. Sazim Mahmudur Rahman** (Reg.: 2019331087 , CSE , SUST)  
- **Pappu Roy** (Reg.: 2019331120, CSE , SUST)  

---

## **Acknowledgment**  
This project has been conducted by the **Department of Computer Science and Engineering, SUST** as part of a **CSE 450** Course.  

---

## **License**  
This project is licensed under the [MIT License](LICENSE).

--- 
