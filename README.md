# Socioeconomic Disparities Across Generations: A Machine Learning Approach

This repository contains the code and data used in the research study: **"Socioeconomic Disparities Across Generations: A Focus on Generation Z"**. The study explores generational differences in socioeconomic features, such as income distribution and housing affordability, using machine learning techniques to identify patterns and predict generational membership and year of birth.

## Overview

### Key Highlights:
- **Purpose**: To analyze socioeconomic disparities across generational cohorts, with a focus on Generation Z, and uncover systemic challenges impacting their financial outcomes.
- **Techniques Used**:
  - Classification models to predict generational membership.
  - Regression models to estimate the year of birth as a continuous variable.
  - Feature selection using mutual information.
  - Explainable Boosting Machine (EBM) as the best-performing classifier.
- **Key Results**:
  - Classification accuracy: **74.62%** (highest achieved by EBM).
  - Regression performance: **$R^2 = 0.6005$** with an average absolute error of **eight years**.
  - Insights: Generation Z experiences the highest rent-to-income burden (60.0%) and faces significant barriers to homeownership.

## Repository Contents

This repository includes:
- **Code**:
  - Scripts for data preprocessing, feature selection, model training, and evaluation.
  - Implementation of classification and regression models.
- **Data**:
  - Processed datasets used for training and evaluation.
  - Metadata and descriptions of features included in the analysis.
- **Documentation**:
  - Detailed explanations of methods, feature selection processes, and model results.
  - Jupyter notebooks for step-by-step analysis and visualization.
- **Results**:
  - Model outputs, performance metrics, and summary statistics.

## Getting Started

### Prerequisites
- Python 3.8+
- Required Python libraries (listed in `requirements.txt`).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/socioeconomic-disparities.git
