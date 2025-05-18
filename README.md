# Hotel Data Analysis & Prediction Dashboard

This project  uses a hotel booking dataset to analyze guest behavior, make predictions, and show everything in an interactive dashboard using Streamlit.

---

## Problem Statement

The goal was to create an interactive platform for hotel management that:

- Shows different metrics like revenue, stay length, etc.
- Segments guests based on behavior
- Predicts future values like occupancy and amenity usage
- Helps visualize and explore hotel trends

---

## Tools and Libraries Used

Some of the libraries I used:

- pandas: for data handling and cleaning
- numpy: for calculations and feature engineering
- matplotlib, seaborn: for graphs
- scikit-learn: for clustering and prediction
- streamlit: for creating the dashboard
- networkx: for visualizing guest journey

---

## Features 

- **REI**: Revenue Efficiency Index  
- **GSI**: Guest Satisfaction Index  
- **LGS**: Loyalty Generation Score  
- **AUR**: Amenity Usage Ratio  
- **Guest Segmentation** using KMeans
- **Revenue by Segment**
- **Occupancy Forecast** by room type
- **Amenity Usage Forecast** using a neural net
- **Interactive Filters** on the dashboard for demographics, seasonality, room type, etc.

---

## How to Run the Dashboard

Step 1: install the required libraries from the requirements.txt file
Step 2: run the following command: streamlit run dashboard.py


