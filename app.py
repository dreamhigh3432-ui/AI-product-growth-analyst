import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Product Growth & Customer Analyst", page_icon="📊", layout="wide")

# -------------------------------
# Load Pre-trained Artifacts
# -------------------------------
@st.cache_resource
def load_churn_artifacts():
    scaler = joblib.load('scaler.pkl')
    kmeans = joblib.load('kmeans.pkl')
    cluster_stats = pd.read_csv('cluster_stats.csv', index_col=0)
    return scaler, kmeans, cluster_stats

scaler, kmeans, cluster_stats = load_churn_artifacts()

# Segment labels from training
segment_labels = cluster_stats['Segment'].to_dict()

# -------------------------------
# Helper Functions
# -------------------------------

def analyze_bundles(df):
    """Market Basket Analysis for product bundles."""
    required_cols = {'Order ID', 'Product'}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must contain 'Order ID' and 'Product' columns.")
    
    basket = df.groupby(['Order ID', 'Product'])['Product'].count().unstack().reset_index().fillna(0)
    basket = basket.set_index('Order ID')
    basket = basket.astype(bool)
    total_orders = basket.shape[0]
    
    if total_orders < 2:
        return [], []
    
    frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
    if frequent_itemsets.empty:
        return [], []
    
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    rules = rules[(rules['antecedents'].apply(lambda x: len(x) == 1)) &
                  (rules['consequents'].apply(lambda x: len(x) == 1))]
    if rules.empty:
        return [], []
    
    rules['pair_key'] = rules.apply(
        lambda row: tuple(sorted([list(row['antecedents'])[0], list(row['consequents'])[0]])), axis=1
    )
    rules = rules.drop_duplicates(subset='pair_key', keep='first')
    top_rules = rules.nlargest(3, 'lift')
    
    bundles = []
    for _, row in top_rules.iterrows():
        ant = list(row['antecedents'])[0]
        cons = list(row['consequents'])[0]
        bundles.append({
            "product_a": ant,
            "product_b": cons,
            "confidence": round(row['confidence'], 2),
            "lift": round(row['lift'], 2),
            "support": round(row['support'], 2),
            "bundle_count": int(row['support'] * total_orders)
        })
    
    simulations = []
    if 'Price' in df.columns:
        product_prices = df.groupby('Product')['Price'].mean().to_dict()
        for b in bundles:
            p_a = product_prices.get(b['product_a'])
            p_b = product_prices.get(b['product_b'])
            if p_a and p_b:
                sim = simulate_discount(
                    f"{b['product_a']} + {b['product_b']}",
                    b['bundle_count'],
                    p_a + p_b
                )
                simulations.append(sim)
    return bundles, simulations

def simulate_discount(bundle_name, current_sales, total_price, elasticity=-1.5):
    """Simulate profit impact of different discount levels."""
    scenarios = []
    baseline_revenue = current_sales * total_price
    for disc in [0.05, 0.10, 0.15, 0.20, 0.25]:
        new_price = total_price * (1 - disc)
        demand_change_pct = elasticity * (-disc)
        new_sales_only = current_sales * demand_change_pct
        total_sales = current_sales + new_sales_only
        revenue_current = current_sales * total_price
        revenue_new = new_sales_only * new_price
        revenue = revenue_current + revenue_new
        discount_cost = disc * total_price * new_sales_only
        extra_rev = revenue - baseline_revenue
        profit_impact = extra_rev - discount_cost
        scenarios.append({
            "discount_percent": disc,
            "estimated_total_sales": round(total_sales, 1),
            "new_sales": round(new_sales_only, 1),
            "revenue": round(revenue, 2),
            "extra_revenue_vs_baseline": round(extra_rev, 2),
            "discount_cost": round(discount_cost, 2),
            "profit_impact": round(profit_impact, 2)
        })
    best = max(scenarios, key=lambda x: x['profit_impact'])
    return {
        "bundle": bundle_name,
        "current_sales": current_sales,
        "total_price": total_price,
        "scenarios": scenarios,
        "best_discount": best
    }

def segment_customers(orders_df):
    """Segment customers using pre-trained K-Means model."""
    required_cols = {'Customer ID', 'Order Date', 'Order ID', 'Total Price'}
    if not required_cols.issubset(orders_df.columns):
        raise ValueError("CSV must contain: Customer ID, Order Date, Order ID, Total Price")
    
    orders_df['Order Date'] = pd.to_datetime(orders_df['Order Date'])
    snapshot = orders_df['Order Date'].max() + timedelta(days=1)
    
    rfm = orders_df.groupby('Customer ID').agg({
        'Order Date': lambda x: (snapshot - x.max()).days,
        'Order ID': 'nunique',
        'Total Price': 'sum'
    }).rename(columns={'Order Date': 'Recency', 'Order ID': 'Frequency', 'Total Price': 'Monetary'})
    
    rfm_scaled = scaler.transform(rfm[['Recency', 'Frequency', 'Monetary']])
    rfm['Cluster'] = kmeans.predict(rfm_scaled)
    rfm['Segment'] = rfm['Cluster'].map(segment_labels)
    
    def recommendation(seg):
        if 'At Risk' in str(seg):
            return "Send win-back email with 20% discount. Highlight new arrivals."
        elif 'Needs Attention' in str(seg):
            return "Offer free shipping on next order. Recommend complementary products."
        elif 'Loyal' in str(seg):
            return "Invite to VIP program. Ask for review or referral."
        elif 'Champions' in str(seg):
            return "Reward with early access or special gift. Encourage social sharing."
        return "No specific action needed."
    
    rfm['Recommendation'] = rfm['Segment'].apply(recommendation)
    return rfm

# -------------------------------
# Main UI
# -------------------------------
st.title("📊 AI Product Growth & Customer Analyst")
st.markdown("Upload your store data to get AI-powered bundle recommendations and customer retention insights.")

# Privacy notice
st.caption("🔒 Your data is processed in memory only — nothing is stored or shared.")

# Tabs
tab1, tab2 = st.tabs(["📦 Bundle & Discount Analyzer", "👥 Customer Segmentation"])

# ==========================================
# TAB 1: Bundle Analyzer
# ==========================================
with tab1:
    st.header("Product Bundle & Discount Optimizer")
    st.markdown("Upload your orders CSV (with `Order ID`, `Product`, and optionally `Price` columns).")
    
    # Sample CSV download
    if os.path.exists('sample_orders.csv'):
        with open('sample_orders.csv', 'rb') as f:
            st.download_button(
                label="📥 Download Sample CSV",
                data=f,
                file_name='sample_orders.csv',
                mime='text/csv'
            )
    
    bundle_file = st.file_uploader("Choose a CSV file", type='csv', key='bundle')
    
    if bundle_file is not None:
        try:
            bundle_df = pd.read_csv(bundle_file)
            st.write("### Data Preview:")
            st.dataframe(bundle_df.head(10), use_container_width=True)
            
            if st.button("🔍 Analyze Bundles & Discount Impact", type="primary"):
                with st.spinner("AI is analyzing..."):
                    bundles, simulations = analyze_bundles(bundle_df)
                    if bundles:
                        st.success(f"✅ Found {len(bundles)} strong bundle(s)!")
                        for b in bundles:
                            with st.container():
                                st.subheader(f"📦 {b['product_a']} + {b['product_b']}")
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Lift", f"{b['lift']}x")
                                c2.metric("Confidence", f"{b['confidence']:.0%}")
                                c3.metric("Current Sales (in data)", b['bundle_count'])
                                
                                sim = next((s for s in simulations if s['bundle'] == f"{b['product_a']} + {b['product_b']}"), None)
                                if sim:
                                    st.markdown("### 🧠 AI Discount Optimizer")
                                    st.caption("⚙️ Simulation based on an assumed price elasticity (1.5). "
                                               "Actual results may vary. We recommend testing on a small scale first.")
                                    best = sim['best_discount']
                                    st.info(f"**Optimal Discount:** {best['discount_percent']:.0%} → "
                                            f"Estimated {best['estimated_total_sales']} sales, "
                                            f"Extra profit impact: ${best['profit_impact']:,.2f}")
                                    if best['discount_percent'] >= 0.20:
                                        st.warning("A discount above 15% may reduce margins or brand perception. Test with caution.")
                                    with st.expander("See all scenarios"):
                                        df_sim = pd.DataFrame(sim['scenarios'])
                                        df_sim['discount_percent'] = df_sim['discount_percent'].apply(lambda x: f"{x:.0%}")
                                        st.dataframe(df_sim, use_container_width=True)
                                else:
                                    st.caption("Add a `Price` column to unlock discount impact predictions.")
                                st.markdown("---")
                    else:
                        st.warning("No strong bundles found. Try with more transaction data.")
        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# TAB 2: Customer Segmentation
# ==========================================
with tab2:
    st.header("Customer Segmentation & Retention")
    st.markdown("Upload your orders CSV (with `Customer ID`, `Order Date`, `Order ID`, `Total Price` columns).")
    st.caption("⚙️ This model was pre-trained on the Online Retail dataset. For best accuracy, it will be fine-tuned on your store data in future versions.")
    
    churn_file = st.file_uploader("Choose a CSV file", type='csv', key='churn')
    
    if churn_file is not None:
        try:
            churn_df = pd.read_csv(churn_file)
            st.write("### Data Preview:")
            st.dataframe(churn_df.head(10), use_container_width=True)
            
            if st.button("🔍 Analyze Customer Segments", type="primary", key='churn_btn'):
                with st.spinner("Segmenting customers..."):
                    try:
                        rfm_results = segment_customers(churn_df)
                        
                        # Segment summary
                        st.subheader("📊 Segment Distribution")
                        seg_counts = rfm_results['Segment'].value_counts().reset_index()
                        seg_counts.columns = ['Segment', 'Customer Count']
                        st.dataframe(seg_counts, use_container_width=True)
                        
                        # At-risk customers
                        at_risk = rfm_results[rfm_results['Segment'].str.contains('At Risk', na=False)].sort_values('Recency', ascending=False)
                        st.subheader(f"🚨 At-Risk Customers ({len(at_risk)})")
                        if len(at_risk) > 0:
                            st.dataframe(at_risk[['Recency', 'Frequency', 'Monetary', 'Recommendation']].head(20), use_container_width=True)
                        else:
                            st.success("No at-risk customers detected!")
                        
                        # Champions
                        champions = rfm_results[rfm_results['Segment'].str.contains('Champions', na=False)]
                        st.subheader(f"🏆 Champions ({len(champions)})")
                        if len(champions) > 0:
                            st.dataframe(champions[['Recency', 'Frequency', 'Monetary', 'Recommendation']].head(10), use_container_width=True)
                        
                        # Loyal
                        loyal = rfm_results[rfm_results['Segment'].str.contains('Loyal', na=False)]
                        st.subheader(f"💎 Loyal Customers ({len(loyal)})")
                        if len(loyal) > 0:
                            st.dataframe(loyal[['Recency', 'Frequency', 'Monetary', 'Recommendation']].head(10), use_container_width=True)
                        
                        # Needs Attention
                        attention = rfm_results[rfm_results['Segment'].str.contains('Needs Attention', na=False)]
                        st.subheader(f"⚠️ Needs Attention ({len(attention)})")
                        if len(attention) > 0:
                            st.dataframe(attention[['Recency', 'Frequency', 'Monetary', 'Recommendation']].head(10), use_container_width=True)
                            
                    except ValueError as e:
                        st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built for Shopify store owners — AI-powered insights from your transaction data.")
