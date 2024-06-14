# Machine-Learning-Algorithms
This repository contaiins all the machine learning algorithms with diverse Projects.

# Boston House Price Prediction using K-Nearest Neighbors (KNN)

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This project aims to predict house prices in Boston using the K-Nearest Neighbors (KNN) algorithm only. The dataset used for this project contains various features of houses in Boston, such as crime rate, number of rooms, age of the house, and more. The goal is to build a regression model that accurately predicts the prices of houses based on these features.

## Dataset

The dataset used in this project is the Boston Housing Dataset, which is publicly available in the `sklearn` library. It includes 14 attributes (13 features and 1 target variable).

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing)
- **Features**:
  - CRIM: Per capita crime rate by town.
  - ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.
  - INDUS: Proportion of non-retail business acres per town.
  - CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
  - NOX: Nitric oxide concentration (parts per 10 million).
  - RM: Average number of rooms per dwelling.
  - AGE: Proportion of owner-occupied units built before 1940.
  - DIS: Weighted distances to five Boston employment centers.
  - RAD: Index of accessibility to radial highways.
  - TAX: Full-value property tax rate per $10,000.
  - PTRATIO: Pupil-teacher ratio by town.
  - B: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town.
  - LSTAT: Percentage of lower status of the population.
  - PRICE: Median value of owner-occupied homes in $1000s (target variable).

## Installation

To get started with this project, follow these steps:

1. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/boston-house-price-prediction.git
   cd boston-house-price-prediction
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Jupyter Notebook:**
   ```
      jupyter notebook
   ```

2. **Open the notebook:**
   Open `Boston_House_Price_Prediction.ipynb` in your Jupyter Notebook interface.

3. **Follow the steps in the notebook to preprocess the data, train the KNN model, and evaluate its performance.**

## Methodology

The project follows these main steps:

1. **Data Preprocessing**:
   - Handling missing values.
   - Standardizing the features.

2. **Exploratory Data Analysis (EDA)**:
   - Visualizing distributions and relationships between features.

3. **Model Implementation**:
   - Splitting the data into training and testing sets.
   - Training the KNN model with the training data.
   - Hyperparameter tuning using cross-validation.

4. **Model Evaluation**:
   - Evaluating the model using metrics such as Mean Squared Error (MSE) and R-squared.

## Results

- The KNN model's performance is evaluated based on various metrics.
- Visualizations of the model's predictions versus actual house prices are provided.
- Insights and potential improvements are discussed.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to submit a pull request or open an issue.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [scikit-learn](https://scikit-learn.org/stable/)
- [Jupyter Notebook](https://jupyter.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- UCI Machine Learning Repository for the Boston Housing Dataset
