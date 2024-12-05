# **Water Quality Detection Project**  
This project focuses on detecting water quality using various machine learning models. It involves data preprocessing, visualization, feature scaling, feature selection, and applying different classification models to determine whether water is safe or not.  

## **Project Overview**  
Water quality is a critical parameter for maintaining human health. This project uses data analysis and machine learning techniques to classify water samples as safe or unsafe based on chemical and biological properties.  

## **Features of the Project**
- **Dataset Analysis**: Understand the data structure, handle null values, and map categorical features.
- **Data Visualization**: Generate scatter plots, histograms, and heatmaps to visualize relationships between features.
- **Preprocessing**: Handle missing values, encode categorical data, and scale numerical features.
- **Feature Selection**: Analyze feature correlations and select relevant numerical features.
- **Model Training and Evaluation**: Train multiple machine learning models to classify water quality:
  - Support Vector Machines (SVM)
  - Linear Regression
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Logistic Regression  

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**: 
  - Data Analysis: `pandas`, `numpy`
  - Data Visualization: `matplotlib`, `seaborn`, `plotly`
  - Machine Learning: `scikit-learn`

## **Setup Instructions**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/water-quality-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd water-quality-detection
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Add the dataset to the project directory (`/content/drive/MyDrive/` or adjust the file path as needed).
5. Run the Jupyter notebook or Python script:
   ```bash
   jupyter notebook Water Quality detection.ipynb
   ```
   Or execute the Python script:
   ```bash
   python water_quality_detection.py
   ```

## **Dataset Details**
- **File**: `water_quality - Sheet1.xlsx`
- **Target Variable**: `is_safe` (mapped to binary values: `Yes=1`, `No=0`)

## **Machine Learning Models and Results**
| **Model**            | **Accuracy** |
|-----------------------|--------------|
| Support Vector Machine (SVM) | 93.31%       |
| Linear Regression     | 83.81%       |
| Decision Tree         | 96.93%       |
| K-Nearest Neighbors (KNN) | 91.00%       |
| Naive Bayes           | 89.81%       |
| Logistic Regression   | 83.81%       |

## **Project Flow**
1. **Data Preprocessing**: Handle missing values, encode categorical data, and scale features.  
2. **Visualization**: Understand data distribution and relationships.  
3. **Training and Evaluation**: Train various models, tune hyperparameters, and evaluate performance using metrics such as accuracy, confusion matrix, and classification report.  
4. **Result Analysis**: Compare model performance to choose the best-performing model.  

## **Usage Example**
The model predicts whether water is safe based on provided feature values:
```python
x = clf.predict([[2.29, 0.51, -0.56, 0.45, -0.93, 1.68, 0.48, -0.20, 1.12, 0.17, -0.86, 0.09, 1.63, 0.58, -0.11, -1.74, -0.75, -0.30, 1.04, 2.19, -0.92]])
if x == [1]:
    print("The water is safe.")
else:
    print("The water is not safe.")
```

## **Contributions**
Contributions are welcome! Feel free to raise issues or submit pull requests.  

## **License**
This project is licensed under the MIT License.  

## **Contact**
For queries or feedback, please contact [your-email@example.com].  
