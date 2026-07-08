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
st.markdown("""
Upload your store's transaction data to get **AI-powered insights** for two critical growth areas:
- **Product Bundles & Discount Optimization** — find which products to bundle and the best discount to maximize profit.
- **Customer Segmentation & Retention** — identify at-risk customers before they leave and get actionable retention tips.
""")

st.caption("🔒 Your data is processed in memory only — nothing is stored or shared.")

# Tabs
tab1, tab2 = st.tabs(["📦 Bundle & Discount Analyzer", "👥 Customer Segmentation"])

# ==========================================
# TAB 1: Bundle Analyzer
# ==========================================
with tab1:
    st.header("📦 Product Bundle & Discount Optimizer")
    st.markdown("""
    **How it works:**
    1. Upload your Shopify orders CSV file.
    2. The AI scans all customer transactions to find products frequently bought together.
    3. For each bundle found, it simulates how different discount levels (5%–25%) impact your profit.
    4. You get a clear, data-backed recommendation — no guesswork needed.
    """)
    
    with st.expander("📋 What CSV format do I need?"):
        st.markdown("""
        Your CSV file should contain these columns:
        - `Order ID` — unique identifier for each order.
        - `Product` — product name or SKU.
        - `Price` (optional) — unit price of the product. If included, you'll unlock the discount simulation.
        
        **Example:**
        | Order ID | Product | Price |
        |----------|---------|-------|
        | 1001 | Coffee Arabica | 12.00 |
        | 1001 | French Press | 25.00 |
        | 1002 | Mug | 8.00 |
        """)
    
    with st.expander("🔒 Privacy & Security"):
        st.markdown("""
        - Your file is processed in memory only.
        - No data is stored, saved, or shared.
        - The file is deleted immediately after analysis.
        """)
    
    # Sample CSV download
    if os.path.exists('sample_orders.csv'):
        with open('sample_orders.csv', 'rb') as f:
            st.download_button(
                label="📥 Download Sample CSV",
                data=f,
                file_name='sample_orders.csv',
                mime='text/csv'
            )
        st.caption("Don't have your own data yet? Download the sample file above to see how the tool works.")
    
    bundle_file = st.file_uploader("Upload your orders CSV file", type='csv', key='bundle')
    
    if bundle_file is not None:
        try:
            bundle_df = pd.read_csv(bundle_file)
            st.write("### Data Preview (first 10 rows):")
            st.dataframe(bundle_df.head(10), use_container_width=True)
            st.caption(f"Total rows loaded: {len(bundle_df):,}")
            
            if st.button("🔍 Analyze Bundles & Discount Impact", type="primary"):
                with st.spinner("AI is analyzing product relationships..."):
                    bundles, simulations = analyze_bundles(bundle_df)
                    if bundles:
                        st.success(f"✅ Found {len(bundles)} strong bundle opportunity(s)!")
                        for b in bundles:
                            with st.container():
                                st.subheader(f"📦 Bundle: {b['product_a']} + {b['product_b']}")
                                
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Lift", f"{b['lift']}x", help="How much more likely these two are bought together vs. random chance. Above 1.2 is significant.")
                                col2.metric("Confidence", f"{b['confidence']:.0%}", help="The probability that a customer who buys product A also buys product B.")
                                col3.metric("Times Bought Together", b['bundle_count'], help="How many orders in your data contain both products.")
                                
                                st.markdown("""
                                > 💡 **What this means:** These products have a strong relationship. Bundling them can increase your average order value.
                                """)
                                
                                sim = next((s for s in simulations if s['bundle'] == f"{b['product_a']} + {b['product_b']}"), None)
                                if sim:
                                    st.markdown("### 🧠 AI Discount Optimizer")
                                    st.markdown("""
                                    The simulation below estimates how different discount levels would impact your profit.
                                    It uses a standard price elasticity assumption (every 1% price drop → 1.5% demand increase).
                                    """)
                                    st.caption("⚙️ Simulation based on an assumed price elasticity (1.5). Actual results may vary. We recommend testing any discount on a small scale first.")
                                    
                                    best = sim['best_discount']
                                    st.info(f"**🏆 Optimal Discount:** {best['discount_percent']:.0%} → "
                                            f"Estimated {best['estimated_total_sales']} sales, "
                                            f"Extra profit impact: ${best['profit_impact']:,.2f}")
                                    
                                    if best['discount_percent'] >= 0.20:
                                        st.warning("⚠️ A discount above 15% may reduce margins or brand perception. Test with caution.")
                                    
                                    with st.expander("📊 See all discount scenarios"):
                                        st.markdown("""
                                        This table shows the estimated impact of each discount level:
                                        - **Estimated Total Sales:** projected number of bundle sales.
                                        - **Revenue:** total revenue from the bundle.
                                        - **Profit Impact:** the additional profit (or loss) compared to not offering a discount.
                                        """)
                                        df_sim = pd.DataFrame(sim['scenarios'])
                                        df_sim['discount_percent'] = df_sim['discount_percent'].apply(lambda x: f"{x:.0%}")
                                        st.dataframe(df_sim, use_container_width=True)
                                else:
                                    st.info("💡 Add a `Price` column to your CSV to unlock the discount impact simulation.")
                                st.markdown("---")
                    else:
                        st.warning("No strong product bundles found in your data. This could mean:")
                        st.markdown("""
                        - Your customers tend to buy single products.
                        - You need more transaction data for the AI to find patterns.
                        - Try lowering the min_support threshold (contact us for a custom analysis).
                        """)
        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# TAB 2: Customer Segmentation
# ==========================================
with tab2:
    st.header("👥 Customer Segmentation & Retention")
    st.markdown("""
    **How it works:**
    1. Upload your order history CSV file with customer-level data.
    2. The AI calculates **RFM scores** for each customer:
       - **Recency:** How recently did they buy?
       - **Frequency:** How often do they buy?
       - **Monetary:** How much do they spend?
    3. Customers are automatically grouped into 4 segments using machine learning.
    4. You get a prioritized list of at-risk customers with **specific retention actions**.
    """)
    
    with st.expander("📋 What CSV format do I need?"):
        st.markdown("""
        Your CSV file should contain these columns:
        - `Customer ID` — unique identifier for each customer.
        - `Order Date` — date of the order (YYYY-MM-DD format recommended).
        - `Order ID` — unique identifier for each order.
        - `Total Price` — total amount of the order.
        
        **Example:**
        | Customer ID | Order Date | Order ID | Total Price |
        |-------------|------------|----------|-------------|
        | 12345 | 2024-01-15 | INV-001 | 25.00 |
        | 12345 | 2024-02-20 | INV-002 | 30.00 |
        | 67890 | 2024-03-10 | INV-003 | 15.00 |
        """)
    
    with st.expander("🔒 Privacy & Security"):
        st.markdown("""
        - Your file is processed in memory only.
        - No customer data is stored, saved, or shared.
        - The file is deleted immediately after analysis.
        """)
    
    with st.expander("🧠 How the AI model works"):
        st.markdown("""
        This tool uses a **K-Means clustering algorithm** trained on the famous Online Retail dataset.
        It groups customers based on their purchasing behavior into these segments:
        
        | Segment | Description |
        |---------|-------------|
        | 🏆 **Champions** | Your best customers — buy recently, frequently, and spend the most. |
        | 💎 **Loyal** | Regular customers who buy consistently. |
        | ⚠️ **Needs Attention** | Starting to drift away — above average recency but lower frequency. |
        | 🚨 **At Risk** | Haven't purchased in a long time — high chance of churning. |
        
        > ⚙️ *Note: This model was pre-trained on a general e-commerce dataset. For optimal accuracy on your store, future versions will fine-tune the model on your specific data.*
        """)
    
    churn_file = st.file_uploader("Upload your orders CSV file", type='csv', key='churn')
    
    if churn_file is not None:
        try:
            churn_df = pd.read_csv(churn_file)
            st.write("### Data Preview (first 10 rows):")
            st.dataframe(churn_df.head(10), use_container_width=True)
            st.caption(f"Total rows loaded: {len(churn_df):,}")
            
            if st.button("🔍 Analyze Customer Segments", type="primary", key='churn_btn'):
                with st.spinner("Segmenting customers..."):
                    try:
                        rfm_results = segment_customers(churn_df)
                        
                        # Segment summary
                        st.subheader("📊 Segment Distribution")
                        st.markdown("Here's how your customer base breaks down into behavioral segments:")
                        seg_counts = rfm_results['Segment'].value_counts().reset_index()
                        seg_counts.columns = ['Segment', 'Customer Count']
                        st.dataframe(seg_counts, use_container_width=True)
                        
                        st.markdown("---")
                        
                        # At-risk customers
                        at_risk = rfm_results[rfm_results['Segment'].str.contains('At Risk', na=False)].sort_values('Recency', ascending=False)
                        st.subheader(f"🚨 At-Risk Customers ({len(at_risk)})")
                        st.markdown("""
                        These customers haven't purchased in a long time and are likely to churn.
                        **Recommended action:** Send them a personalized win-back email with a compelling offer.
                        """)
                        if len(at_risk) > 0:
                            st.dataframe(at_risk[['Recency', 'Frequency', 'Monetary', 'Recommendation']].head(20), use_container_width=True)
                        else:
                            st.success("Great news — no at-risk customers detected!")
                        
                        st.markdown("---")
                        
                        # Champions
                        champions = rfm_results[rfm_results['Segment'].str.contains('Champions', na=False)]
                        st.subheader(f"🏆 Champions ({len(champions)})")
                        st.markdown("""
                        Your top customers — they buy recently, frequently, and spend the most.
                        **Recommended action:** Reward them with exclusive perks, early access, or a loyalty gift. They're your brand advocates.
                        """)
                        if len(champions) > 0:
                            st.dataframe(champions[['Recency', 'Frequency', 'Monetary', 'Recommendation']].head(10), use_container_width=True)
                        
                        st.markdown("---")
                        
                        # Loyal
                        loyal = rfm_results[rfm_results['Segment'].str.contains('Loyal', na=False)]
                        st.subheader(f"💎 Loyal Customers ({len(loyal)})")
                        st.markdown("""
                        Solid, consistent buyers who trust your store.
                        **Recommended action:** Invite them to a VIP program, ask for reviews, or upsell premium products.
                        """)
                        if len(loyal) > 0:
                            st.dataframe(loyal[['Recency', 'Frequency', 'Monetary', 'Recommendation']].head(10), use_container_width=True)
                        
                        st.markdown("---")
                        
                        # Needs Attention
                        attention = rfm_results[rfm_results['Segment'].str.contains('Needs Attention', na=False)]
                        st.subheader(f"⚠️ Needs Attention ({len(attention)})")
                        st.markdown("""
                        These customers are starting to drift. They don't buy as often as your loyal customers.
                        **Recommended action:** Send a friendly re-engagement offer — free shipping or a small discount can bring them back.
                        """)
                        if len(attention) > 0:
                            st.dataframe(attention[['Recency', 'Frequency', 'Monetary', 'Recommendation']].head(10), use_container_width=True)
                        
                        st.markdown("---")
                        st.success("""
                        ✅ **Analysis complete!** Use these insights to:
                        - Send targeted retention campaigns to at-risk customers.
                        - Reward your Champions to keep them loyal.
                        - Re-engage the "Needs Attention" group before they slip away.
                        """)
                            
                    except ValueError as e:
                        st.error(f"Error: {e}")
                        st.info("Make sure your CSV has these columns: Customer ID, Order Date, Order ID, Total Price")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("""
### 🚀 What's Coming Next?
- **Direct Shopify Integration** — connect your store with one click, no CSV uploads needed.
- **Weekly Automated Reports** — get AI insights delivered to your inbox every week.
- **Fine-Tuned Models** — the AI will learn your specific customer patterns for even better predictions.
""")
st.caption("Built for Shopify store owners — AI-powered insights from your transaction data.")
