import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from typing import Dict, Any

st.set_page_config(page_title="AI Product Growth Analyst", page_icon="📊", layout="wide")

st.title("📊 AI Product Growth Analyst")
st.subheader("Bundle Recommendations for Your Shopify Store")

st.markdown("""
Upload your Shopify orders CSV to get **AI-powered bundle suggestions** and see how discounts could boost your sales.  
For best results, include a `Price` column (Lineitem price) in your export.
""")

with st.expander("🔒 Privacy & Security"):
    st.markdown("""
    - Your file is processed in memory – no data is stored.
    - We only use `Order ID`, `Product`, and optionally `Price` columns.
    - No customer information is kept or shared.
    """)

# -------------------------------
# Core analysis functions (same as API)
# -------------------------------

def analyze_bundles(df: pd.DataFrame):
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
    rules = rules[(rules['antecedents'].apply(lambda x: len(x)==1)) &
                  (rules['consequents'].apply(lambda x: len(x)==1))]
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

# -------------------------------
# Streamlit UI
# -------------------------------

# Sample CSV download button
with open('sample_orders.csv', 'rb') as f:
    st.download_button(
        label="📥 Download Sample CSV",
        data=f,
        file_name='sample_orders.csv',
        mime='text/csv'
    )

st.caption("🔒 Your file is processed in memory only — no data is stored or shared.")
uploaded_file = st.file_uploader("Drop your orders CSV here", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview:")
        st.dataframe(df.head(10), use_container_width=True)
        
        if st.button("🔍 Analyze Bundles & Discount Impact", type="primary"):
            with st.spinner("AI is analyzing..."):
                bundles, simulations = analyze_bundles(df)
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
                                st.markdown("#### 💰 Discount Impact Simulation")
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
        st.error(f"Could not read CSV: {e}")

st.markdown("---")
st.caption("Built for Shopify store owners – from CSV to revenue-boosting bundles in seconds.")
