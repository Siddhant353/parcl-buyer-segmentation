import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Buyer Segmentation Dashboard", layout="wide")

# -----------------------------
# Data Loading
# -----------------------------
@st.cache_data

def load_data():
    df = pd.read_csv("../Data-processed/clients_clustered_named.csv")
    return df


df = load_data()

# -----------------------------
# Title and Intro
# -----------------------------
st.title("Buyer Segmentation and Investment Profiling Dashboard")
st.markdown(
    "This dashboard presents buyer segments created using machine learning-based clustering. "
    "Use the filters on the left to explore differences in buyer behavior, geography, financing, and property characteristics."
)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

country_options = ["All"] + sorted(df["country"].dropna().unique().tolist())
region_options = ["All"] + sorted(df["region"].dropna().unique().tolist())
purpose_options = ["All"] + sorted(df["acquisition_purpose"].dropna().unique().tolist())
client_type_options = ["All"] + sorted(df["client_type"].dropna().unique().tolist())

selected_country = st.sidebar.selectbox("Country", country_options)
selected_region = st.sidebar.selectbox("Region", region_options)
selected_purpose = st.sidebar.selectbox("Acquisition Purpose", purpose_options)
selected_client_type = st.sidebar.selectbox("Client Type", client_type_options)

filtered_df = df.copy()

if selected_country != "All":
    filtered_df = filtered_df[filtered_df["country"] == selected_country]

if selected_region != "All":
    filtered_df = filtered_df[filtered_df["region"] == selected_region]

if selected_purpose != "All":
    filtered_df = filtered_df[filtered_df["acquisition_purpose"] == selected_purpose]

if selected_client_type != "All":
    filtered_df = filtered_df[filtered_df["client_type"] == selected_client_type]

# -----------------------------
# Safety Check
# -----------------------------
if filtered_df.empty:
    st.warning("No records match the selected filters. Please adjust the filter settings.")
    st.stop()

# -----------------------------
# Key Metrics
# -----------------------------
st.subheader("Dashboard Summary")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Clients", len(filtered_df))
col2.metric("Clusters Present", filtered_df["cluster_name"].nunique())
col3.metric("Average Age", round(filtered_df["age"].mean(), 2))
col4.metric("Average Satisfaction", round(filtered_df["satisfaction_score"].mean(), 2))

# -----------------------------
# Buyer Segmentation Overview
# -----------------------------
st.header("1. Buyer Segmentation Overview")

cluster_counts = filtered_df["cluster_name"].value_counts().reset_index()
cluster_counts.columns = ["cluster_name", "client_count"]

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(cluster_counts["cluster_name"], cluster_counts["client_count"])
ax1.set_title("Cluster Distribution")
ax1.set_xlabel("Cluster Name")
ax1.set_ylabel("Number of Clients")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
st.pyplot(fig1)

st.dataframe(cluster_counts, width="stretch")

# -----------------------------
# Investor Behavior Dashboard
# -----------------------------
st.header("2. Investor Behavior Dashboard")

col_a, col_b = st.columns(2)

with col_a:
    purpose_by_cluster = pd.crosstab(filtered_df["cluster_name"], filtered_df["acquisition_purpose"])
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    purpose_by_cluster.plot(kind="bar", ax=ax2)
    ax2.set_title("Acquisition Purpose by Cluster")
    ax2.set_xlabel("Cluster Name")
    ax2.set_ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    st.pyplot(fig2)

with col_b:
    loan_by_cluster = pd.crosstab(filtered_df["cluster_name"], filtered_df["loan_applied"])
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    loan_by_cluster.plot(kind="bar", ax=ax3)
    ax3.set_title("Loan Status by Cluster")
    ax3.set_xlabel("Cluster Name")
    ax3.set_ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    st.pyplot(fig3)

satisfaction_by_cluster = filtered_df.groupby("cluster_name")["satisfaction_score"].mean().sort_values(ascending=False)
fig4, ax4 = plt.subplots(figsize=(10, 5))
ax4.bar(satisfaction_by_cluster.index, satisfaction_by_cluster.values)
ax4.set_title("Average Satisfaction Score by Cluster")
ax4.set_xlabel("Cluster Name")
ax4.set_ylabel("Average Satisfaction Score")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
st.pyplot(fig4)

# -----------------------------
# Geographic Buyer Analysis
# -----------------------------
st.header("3. Geographic Buyer Analysis")

col_c, col_d = st.columns(2)

with col_c:
    country_counts = filtered_df["country"].value_counts().head(10)
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    ax5.bar(country_counts.index, country_counts.values)
    ax5.set_title("Top Countries in Filtered Dataset")
    ax5.set_xlabel("Country")
    ax5.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig5)

with col_d:
    region_counts = filtered_df["region"].value_counts().head(10)
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    ax6.bar(region_counts.index, region_counts.values)
    ax6.set_title("Top Regions in Filtered Dataset")
    ax6.set_xlabel("Region")
    ax6.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig6)

geo_cluster = pd.crosstab(filtered_df["cluster_name"], filtered_df["country"])
st.subheader("Country Distribution by Cluster")
st.dataframe(geo_cluster, width="stretch")

# -----------------------------
# Segment Insights Panel
# -----------------------------
st.header("4. Segment Insights Panel")

segment_summary = (
    filtered_df.groupby("cluster_name")
    .agg(
        client_count=("client_id", "count"),
        avg_age=("age", "mean"),
        avg_satisfaction=("satisfaction_score", "mean"),
        avg_sold_property_count=("sold_property_count", "mean"),
        avg_sale_price=("avg_sale_price", "mean"),
        avg_floor_area_sqft=("avg_floor_area_sqft", "mean")
    )
    .round(2)
    .reset_index()
)

st.dataframe(segment_summary, width="stretch")

cluster_descriptions = {
    "Premium Large-Property Buyers": "Higher-value buyers associated with larger floor-area properties and stronger premium-market positioning.",
    "Older Multi-Property Buyers": "A small but distinct older segment with the highest transaction intensity and repeat or portfolio-style behavior.",
    "Dissatisfied Mid-Value Buyers": "Mid-value buyers whose most distinctive feature is weak satisfaction, suggesting service or journey friction.",
    "Satisfied Value-Oriented Buyers": "The largest cluster, characterized by lower-value properties but stronger satisfaction and stable mainstream behavior."
}

selected_cluster_name = st.selectbox(
    "Select a cluster to view its business interpretation",
    sorted(filtered_df["cluster_name"].unique().tolist())
)

st.markdown(f"**Cluster Description:** {cluster_descriptions.get(selected_cluster_name, 'Description not available.')}" )

selected_cluster_df = filtered_df[filtered_df["cluster_name"] == selected_cluster_name]

col_e, col_f = st.columns(2)
with col_e:
    st.write("**Top Countries in Selected Cluster**")
    st.dataframe(
        selected_cluster_df["country"].value_counts(normalize=True).mul(100).round(2).head(10).rename("percentage").reset_index(),
        width="stretch",
    )

with col_f:
    st.write("**Top Regions in Selected Cluster**")
    st.dataframe(
        selected_cluster_df["region"].value_counts(normalize=True).mul(100).round(2).head(10).rename("percentage").reset_index(),
        width="stretch",
    )

# -----------------------------
# Data Preview
# -----------------------------
st.header("5. Filtered Data Preview")

preview_rows = st.selectbox("Rows to preview", [10, 25, 50, 100, 500, "All"], index=2)

if preview_rows == "All":
    st.dataframe(filtered_df, width="stretch")
else:
    st.dataframe(filtered_df.head(preview_rows), width="stretch")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name="filtered_clients.csv",
    mime="text/csv"
)
