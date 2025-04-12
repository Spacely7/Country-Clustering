# Country-Clustering
# 🌍 Country Clustering Using K-Means and PCA
This interactive web application was built using Streamlit to help categorize countries based on socio-economic and health indicators. The project simulates a real-world scenario where an NGO, HELP International, aims to identify countries most in need of aid. With limited resources (a $10 million fund), it's crucial to make data-driven decisions to maximize impact.

# 🔍 Objective
The main goal is to cluster countries into groups based on key development indicators. These groupings help the NGO prioritize where to direct their aid and interventions.

# 💡 Key Features of the App
Project Overview
An introduction that outlines the mission of HELP International and the purpose of the analysis.

# Dataset Loading and Exploration

Load and preview the original dataset.

Display selected attributes relevant to development (e.g., child mortality, income, life expectancy).

Automatically scale the data for better clustering results.

# Exploratory Data Analysis (EDA)

Visualize attribute relationships using heatmaps, pairplots, boxplots, and scatterplots.

Analyze statistical metrics like skewness, kurtosis, IQR, and trends across countries.

Gain insight into the distribution and spread of each indicator.

# K-Means Clustering

Apply the K-Means algorithm to segment countries into clusters.

Users can choose the number of clusters.

Visualize the results through cluster plots (e.g., income vs inflation, exports vs imports).

Evaluate clustering performance using the Silhouette Score.

# Hierarchical Clustering

Apply Agglomerative Clustering and Bisecting K-Means.

Visualize the dendrogram to see how countries are hierarchically grouped.

Compare cluster labels with K-Means for consistency.

# Principal Component Analysis (PCA)

A placeholder for implementing PCA to reduce data dimensionality and visualize clusters in 2D space.

Association Rules Mining (Coming Soon)

Future extension to discover relationships between attributes and country characteristics.

# 🛠️ Tools and Technologies Used
Python

Pandas & NumPy – Data loading and transformation

Matplotlib & Seaborn – Data visualization

Scikit-learn – Clustering algorithms and evaluation metrics

SciPy – Hierarchical clustering and dendrograms

Streamlit – For creating the interactive dashboard

This project demonstrates how unsupervised learning techniques like clustering can guide real-world decision-making in global development contexts. It’s a great example of turning raw data into actionable insight using machine learning and data visualization.
