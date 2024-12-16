## **Customer Segmentation Analysis Report**
 ### **Introduction**
In a rapidly evolving business world, understanding customer behavior is critical for success. However, not all customers are the same—some are loyal and high-spending, while others may be infrequent or disengaged. To address this diversity, businesses need a data-driven approach to segment customers and tailor strategies to their specific needs.  

This project tackles customer segmentation by analyzing a **real-world retail dataset** with over **1 million transactions**, using **RFM Analysis** (Recency, Frequency, and Monetary value) and **K-Means Clustering**. The result? Clear, actionable insights that empower businesses to drive engagement, optimize marketing strategies, and increase revenue.  

**Goal of the Analysis**

To segment customers into meaningful groups and answer key business questions:  
1. Who are the most valuable customers?  
2. How can we re-engage inactive customers?  
3. Which groups should be targeted for upselling or loyalty programs?

### **Dataset Overview**
- **Source:** The data is from a UK-based online retailer.  
- **Scope:** Covers two years of transactional data(**2009–2011**).  
- **Size:** Over **1,067,371** transactions, covering various customer purchase behaviors.  
- **Features:** **8 variables**, including transactional details, customer attributes, and product information.

#### **Data Dictionary:**

| **Feature**     | **Description**                              | **Type**            |
|------------------|----------------------------------------------|---------------------|
| **InvoiceNo**    | Unique transaction ID (C-prefix for cancellations). | Categorical         |
| **StockCode**    | Unique product ID.                          | Categorical         |
| **Description**  | Product name.                               | Textual             |
| **Quantity**     | Number of units purchased.                  | Numeric (Real)      |
| **InvoiceDate**  | Transaction date/time.                      | Time-series         |
| **UnitPrice**    | Price per product (£).                      | Numeric (Real)      |
| **CustomerID**   | Unique customer ID.                         | Categorical         |
| **Country**      | Customer's country of residence.            | Categorical         |


### **Methodology**

#### **1. Data Cleaning and Preparation**
- **Handling Missing Data:** Removed rows with null `CustomerID` values.  
- **Filtering Cancellations:** Excluded transactions labeled as canceled (`InvoiceNo` starting with 'C').  
- **Outlier Detection:** Capped anomalies in `Quantity` and `UnitPrice` to maintain robust analysis.  

**Code Snippet: Handling Missing Data and Filtering Cancellations**
```python
# Remove missing CustomerIDs and filter out cancellations
df = df[df['CustomerID'].notnull()]
df = df[~df['InvoiceNo'].str.startswith('C')]

# Remove outliers
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
```

#### **2. Feature Engineering**
**Computed RFM Metrics:**  
- **Recency (R):** Days since the last transaction.  
- **Frequency (F):** Number of transactions per customer.  
- **Monetary Value (M):** Total revenue contributed by each customer.  

**Code Snippet: RFM Calculation**
```python
# Calculate Recency, Frequency, and Monetary metrics
import datetime as dt
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'count',  # Frequency
    'TotalSpend': 'sum'  # Monetary Value (TotalSpend = Quantity * UnitPrice)
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSpend': 'Monetary'})
```

#### **3. K-Means Clustering**
- **Feature Scaling:** Standardized RFM metrics using **Z-Score Scaling** to ensure equal contribution to clustering.  
- **Optimal Cluster Selection:**  
  - **Elbow Method:** Determined the "elbow point" at **k = 4** clusters.  
  - **Silhouette Analysis:** Verified clustering quality and distinctness for k = 2–4.  
- **Cluster Assignment:** Grouped customers into distinct segments for profiling.

**Code Snippet: K-Means Clustering**
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Standardize RFM values
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
```


### **Visualizations**
1. **Elbow Method Plot:** Shows the optimal number of clusters at **k = 4**.  
2. **Silhouette Scores Plot:** Highlights clustering quality and separation.  
3. **Violin Plots:** Illustrate the distribution of recency, frequency, and monetary values within each cluster.  
4. **Cluster Heatmap:** Provides an at-a-glance view of customer segmentation metrics.

**Code Snippet: Visualizing Clusters with Violin Plots**
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Violin plots for cluster profiling
sns.violinplot(x='Cluster', y='Recency', data=rfm)
plt.title("Recency by Cluster")
plt.show()

sns.violinplot(x='Cluster', y='Frequency', data=rfm)
plt.title("Frequency by Cluster")
plt.show()

sns.violinplot(x='Cluster', y='Monetary', data=rfm)
plt.title("Monetary Value by Cluster")
plt.show()
```
### **Key Insights**
1. **Largest Clusters:**
   - **REWARD (~1900 customers):** Semi-active, high recency, low frequency, and low spending.  
   - **RE-ENGAGE (~1700 customers):** Long-inactive, low engagement, minimal spending.  

2. **High-Value Clusters:**
   - **DELIGHT:** High frequency and spending; most engaged and valuable customers.  
   - **UPSELL:** Recent, frequent, and high-spending customers; ideal for upselling.  

3. **Mid-Value Clusters:**
   - **NURTURE:** Moderate recency and spending; potential for growth.  
   - **RETAIN:** Actively engaged with moderate frequency and spending; needs retention efforts.  

4. **Low-Value Cluster:**
   - **PAMPER:** Low engagement but moderate spending; offers growth potential via upselling.

### **Recommendations**
1. **RE-ENGAGE & REWARD:**  
   - Reactivate inactive and semi-active customers with win-back campaigns, discounts, and reminders.

2. **DELIGHT & UPSELL:**  
   - Retain loyal, high-spending customers through exclusive rewards, VIP perks, and personalized offers.  

3. **NURTURE & PAMPER:**  
   - Encourage activity with targeted promotions, upselling strategies, and loyalty incentives.  

4. **RETAIN:**  
   - Prevent churn by maintaining engagement with regular, personalized communication.

### **Action Plan**
1. **For High-Value Clusters (DELIGHT & UPSELL):**  
   - Offer VIP programs, personalized product recommendations, and premium access events to maximize lifetime value.  

2. **For Inactive Clusters (REWARD & RE-ENGAGE):**  
   - Deploy retargeting ads, personalized “We Miss You” campaigns, and discounts to reactivate these groups.  

3. **For Mid-Value and Low-Value Clusters (NURTURE, RETAIN, PAMPER):**  
   - Promote affordable bundles, referral programs, and incentives to drive repeat purchases and higher spending.


### **Conclusion**
This analysis provides a roadmap for targeted marketing by segmenting customers into actionable groups. The results help businesses allocate resources effectively to maximize retention, reactivation, and growth opportunities.  
- **High-value clusters (DELIGHT & UPSELL)** drive significant revenue and require loyalty programs to sustain engagement.  
- **Inactive clusters (REWARD & RE-ENGAGE)** represent untapped opportunities that can be reactivated with strategic campaigns.  
- **Growth clusters (NURTURE & RETAIN)** offer potential for increased spending with the right incentives.   

### **Key Takeaways**
1. **Retention Opportunities:** Prioritize high-value customers with VIP perks and tailored offers.  
2. **Reactivation Potential:** Target inactive customers with win-back campaigns and discounts.  
3. **Growth Potential:** Nurture mid-value clusters to increase spending and engagement.  
4. **Upselling Opportunities:** Unlock untapped revenue from low-frequency but moderate-spending customers.  

#### **Explore the Full Analysis**
- **📂 Full Jupyter Notebook:** [Link Here](#)  
- **📊 Interactive Visualizations:** [Link Here](#)  