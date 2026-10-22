import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="CSV Explorer", layout="wide")

st.title("CSV Explorer")

st.info(
    "**What is this model for, and what if it's wrong?** "
    "The model predicts film performance from movies.csv (found in GitHub repository) in terms of audience, critical, and financial reception. If it is wrong then films that are unsuccessful in their desired categories may be financed."
)

# 1. Upload
st.header("1. Upload data")
uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is None:
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read that file: {e}")
    st.stop()

numeric_cols = df.select_dtypes("number").columns.tolist()

st.dataframe(df.head(20), width="stretch")

if len(numeric_cols) < 3:
    st.warning("This app needs a CSV with at least three numeric columns.")
    st.stop()

# 2. Charts
st.header("2. Chart a column")
cc1, cc2, cc3 = st.columns(3)
chart_type = cc1.selectbox("Chart type", ["Histogram", "Line", "Scatter"])
y_col = cc2.selectbox("Column", numeric_cols, index=min(1, len(numeric_cols) - 1))

if chart_type == "Histogram":
    bins = cc3.slider("Bins", 5, 100, 30)
    fig = px.histogram(df, x=y_col, nbins=bins)
elif chart_type == "Line":
    x_options = ["(row index)"] + [c for c in numeric_cols if c != y_col]
    x_col = cc3.selectbox("X axis", x_options)
    if x_col == "(row index)":
        fig = px.line(df.reset_index(), x="index", y=y_col)
    else:
        agg = df.groupby(x_col, as_index=False)[y_col].mean().sort_values(x_col)
        fig = px.line(agg, x=x_col, y=y_col, markers=True, title=f"Mean {y_col} by {x_col}")
else:
    x_col = cc3.selectbox("X axis", [c for c in numeric_cols if c != y_col])
    fig = px.scatter(df, x=x_col, y=y_col, opacity=0.6)

fig.update_layout(height=450, margin=dict(t=40, l=10, r=10, b=10))
st.plotly_chart(fig, width="stretch")

# 3. Regression
st.header("3. Linear regression")
rc1, rc2 = st.columns(2)
default_target = "lifeExp" if "lifeExp" in numeric_cols else numeric_cols[-1]
target = rc1.selectbox("Target", numeric_cols, index=numeric_cols.index(default_target))
candidates = [c for c in numeric_cols if c != target]
default_inputs = [c for c in ["gdpPercap", "year"] if c in candidates] or candidates[:2]
inputs = rc2.multiselect("Inputs", candidates, default=default_inputs, max_selections=2)

if len(inputs) != 2:
    st.info("Pick two input columns.")
    st.stop()

data = df[inputs + [target]].dropna()
if len(data) < 10:
    st.warning("Not enough complete rows to fit a model.")
    st.stop()

X, y = data[inputs], data[target]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_tr, y_tr)
pred_te = model.predict(X_te)

m1, m2, m3 = st.columns(3)
m1.metric("R²", f"{r2_score(y_te, pred_te):.3f}")
m2.metric("Mean absolute error", f"{mean_absolute_error(y_te, pred_te):,.3g}")
m3.metric("Training rows", f"{len(X_tr):,}")

st.subheader("Prediction Tool")
pc = st.columns(2)
vals = [
    pc[i].number_input(col, value=float(data[col].median()), format="%.4f")
    for i, col in enumerate(inputs)
]
pred = model.predict(pd.DataFrame([vals], columns=inputs))[0]
st.success(f"Predicted **{target}**: **{pred:,.3f}**")

for col, v in zip(inputs, vals):
    if not data[col].min() <= v <= data[col].max():
        st.warning(f"{col} = {v:,.4g} is outside the data range "
                   f"({data[col].min():,.4g} to {data[col].max():,.4g}); this is an extrapolation.")