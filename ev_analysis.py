# ✅ IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ✅ STEP 1: Load Data (Make sure your CSV includes the required columns)
# If your file is different, change 'ev_clustered_data.csv' below
df = pd.read_csv("ev_clustered_data.csv")

# ✅ STEP 2: Make sure 'Cluster' column exists
# If not, run KMeans clustering here (example below):
# from sklearn.cluster import KMeans
# features = df[['Total_EV_Sales_2023', 'Growth_Rate_2023', 'GDP_Per_Capita_INR', 'EV_Per_Capita']]
# kmeans = KMeans(n_clusters=3, random_state=42)
# df['Cluster'] = kmeans.fit_predict(features)

# ✅ STEP 3: Count Clusters and Set Colors
optimal_k = df['Cluster'].nunique()
colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))

# ✅ FIGURE 1: Cluster Subplots
fig1, axs = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Sales vs Growth Rate
for i, cluster_id in enumerate(sorted(df['Cluster'].unique())):
    cluster_data = df[df['Cluster'] == cluster_id]
    axs[0, 0].scatter(cluster_data['Total_EV_Sales_2023'], cluster_data['Growth_Rate_2023'],
                      s=120, c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.7, edgecolors='black')
    for _, row in cluster_data.iterrows():
        axs[0, 0].annotate(row['State'], (row['Total_EV_Sales_2023'], row['Growth_Rate_2023']),
                           textcoords='offset points', xytext=(5, 5), fontsize=9)

axs[0, 0].set_title('EV Market Clusters: Sales vs Growth Rate', fontsize=14, fontweight='bold')
axs[0, 0].set_xlabel('Total EV Sales (2023)', fontweight='bold')
axs[0, 0].set_ylabel('Growth Rate (%)', fontweight='bold')
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=0.3)

# Plot 2: GDP vs Two-Wheeler %
for i, cluster_id in enumerate(sorted(df['Cluster'].unique())):
    cluster_data = df[df['Cluster'] == cluster_id]
    axs[0, 1].scatter(cluster_data['GDP_Per_Capita_INR'], cluster_data['TwoWheeler_Percent'],
                      s=120, c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.7, edgecolors='black')

axs[0, 1].set_title('GDP vs Two-Wheeler Preference', fontsize=14, fontweight='bold')
axs[0, 1].set_xlabel('GDP Per Capita (INR)', fontweight='bold')
axs[0, 1].set_ylabel('Two Wheeler Percentage (%)', fontweight='bold')
axs[0, 1].legend()
axs[0, 1].grid(True, alpha=0.3)

# Plot 3: EV Per Capita vs Total Sales
for i, cluster_id in enumerate(sorted(df['Cluster'].unique())):
    cluster_data = df[df['Cluster'] == cluster_id]
    axs[1, 0].scatter(cluster_data['EV_Per_Capita'], cluster_data['Total_EV_Sales_2023'],
                      s=120, c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.7, edgecolors='black')

axs[1, 0].set_title('EV Per Capita vs Total Sales', fontsize=14, fontweight='bold')
axs[1, 0].set_xlabel('EV Per Capita', fontweight='bold')
axs[1, 0].set_ylabel('Total EV Sales (2023)', fontweight='bold')
axs[1, 0].legend()
axs[1, 0].grid(True, alpha=0.3)

# Plot 4: States per Cluster
cluster_counts = df['Cluster'].value_counts().sort_index()
bars = axs[1, 1].bar(cluster_counts.index, cluster_counts.values,
                     color=[colors[i] for i in range(len(cluster_counts))], alpha=0.9)
axs[1, 1].set_title('States per Cluster', fontsize=14, fontweight='bold')
axs[1, 1].set_xlabel('Cluster', fontweight='bold')
axs[1, 1].set_ylabel('Number of States', fontweight='bold')

for bar in bars:
    height = bar.get_height()
    axs[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}',
                  ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
fig1.savefig("ev_cluster_figure1.png", dpi=300, bbox_inches='tight')
plt.show()

# ✅ FIGURE 2: Vehicle Type & Growth Analysis
fig2, axs2 = plt.subplots(1, 2, figsize=(16, 6))

# Bar Plot: Vehicle Type Share
cluster_means = df.groupby('Cluster')[['TwoWheeler_Percent', 'ThreeWheeler_Percent', 'FourWheeler_Percent']].mean()
cluster_means.plot(kind='bar', ax=axs2[0],
                   color=['#FF9999', '#66B2FF', '#99FF99'], edgecolor='black')
axs2[0].set_title('Vehicle Type Distribution by Cluster', fontsize=14, fontweight='bold')
axs2[0].set_xlabel('Cluster', fontweight='bold')
axs2[0].set_ylabel('Percentage (%)', fontweight='bold')
axs2[0].legend(['2-Wheeler', '3-Wheeler', '4-Wheeler'])
axs2[0].grid(True, alpha=0.3)

# Box Plot: Growth Rate by Cluster
df.boxplot(column='Growth_Rate_2023', by='Cluster', ax=axs2[1])
axs2[1].set_title('Growth Rate by Cluster', fontsize=14, fontweight='bold')
axs2[1].set_xlabel('Cluster', fontweight='bold')
axs2[1].set_ylabel('Growth Rate (%)', fontweight='bold')
axs2[1].grid(True, alpha=0.3)
plt.suptitle('')

plt.tight_layout()
fig2.savefig("ev_cluster_figure2.png", dpi=300, bbox_inches='tight')
plt.show()
