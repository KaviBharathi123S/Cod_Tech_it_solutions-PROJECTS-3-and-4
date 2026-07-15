
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
# Custom CSS for background color
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #d7e7e1;
}

[data-testid="stHeader"] {
    background-color: #d9bede;
}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title
st.title("Customer Segmentation Web App")

# Upload CSV file
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Select features (Age & Spending Score for clustering)
    X = df.iloc[:, [3, 4]].values

    # Elbow Method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    st.subheader("📉 Elbow Method")
    fig, ax = plt.subplots()
    plt.plot(range(1, 11), wcss, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    st.pyplot(fig)

    # Slider for choosing number of clusters
    n_clusters = st.slider("🔢 Select number of clusters", min_value=2, max_value=10, value=5)

    # Final clustering
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Add cluster column to dataset
    df["Cluster"] = y_kmeans
    st.subheader("📌 Dataset with Cluster Labels")
    st.dataframe(df.head())
     # Bar Chart (Cluster Summary)
    st.subheader(f"📊 Average Values per Cluster (Bar Chart, k = {n_clusters})")
    cluster_summary = df.groupby("Cluster")[["Age", "Spending Score (1-100)"]].mean()
    fig3, ax3 = plt.subplots()
    cluster_summary.plot(kind="bar", ax=ax3)
    plt.title("Age  & Spending by Cluster")
    plt.ylabel("Value")
    st.pyplot(fig3)
else:
    st.info("👆 Please upload a CSV file to start.")


