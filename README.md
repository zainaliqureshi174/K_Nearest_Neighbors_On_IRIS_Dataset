# K_Nearest_Neighbors_On_IRIS_Dataset From Scratch
This project implements a K-Nearest Neighbors (KNN) classifier entirely from scratch using Python. It demonstrates the core concepts of the KNN algorithm, leveraging NumPy for numerical operations, Pandas for data handling, and Matplotlib for visualization.

**📚 Overview**
The KNN algorithm is a simple, non-parametric method used for classification and regression. This project provides a step-by-step implementation of KNN, focusing on key functionalities such as:

**Data preprocessing**

Distance calculation
Model training
Classification and performance evaluation
Data visualization

**⚙️ Dependencies**
Ensure you have the following Python libraries installed:

NumPy: For efficient numerical computations
Pandas: For data manipulation and analysis
Matplotlib: For data visualization and plotting
You can install these packages using:

pip install numpy pandas matplotlib

**📂 Project Structure**
KNN_Classifier/
│
├── data/                          # Directory for datasets
│   └── dataset.csv                # Example dataset used for classification
│
├── knn.py                         # Main KNN implementation script
├── utils.py                       # Utility functions (e.g., data loading, scaling)
│
├── requirements.txt               # List of dependencies
└── README.md                      # Project documentation (this file)

**🛠️ Implementation Details**
1. Data Preprocessing
Loading datasets using Pandas
Normalizing features for consistent distance calculations
Splitting the dataset into training and test sets
2. KNN Algorithm Implementation
Distance Calculation: Euclidean distance between data points.
Training: Storing the training data and labels.
Prediction: Identifying the K-nearest neighbors and performing majority voting.
3. Evaluation Metrics
Accuracy calculation
Confusion matrix for detailed performance analysis
4. Visualization
Data distribution and decision boundaries visualized using Matplotlib

**🚀 How to Run the Project**
Clone the repository:

git clone https://github.com/yourusername/KNN_Classifier.git

cd KNN_Classifier

**Run the KNN classifier:**
python knn.py
Customize parameters: Modify the k value or dataset path in knn.py to experiment with different settings.

**📊 Example Usage**

from knn import KNearestNeighbor

# Initialize the KNN classifier
knn = KNearestNeighbor(k=5)

# Train the model
knn.train(X_train, y_train)

# Predict on test data
predictions = knn.predict(X_test)

# Evaluate accuracy
accuracy = knn.evaluate(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

**📈 Results and Visualizations**
Performance metrics, including accuracy and confusion matrices, are printed in the console.
Matplotlib visualizations illustrate decision boundaries and data distributions for better understanding.

**🧩 Future Improvements**
Implementing other distance metrics (Manhattan, Minkowski)
Adding cross-validation for optimal k selection
Extending to handle regression tasks

**🤝 Contributing**
Contributions are welcome! Feel free to fork this repository, open issues, or submit pull requests with improvements or bug fixes.

**Author:**
Muhammad Zain Ali
