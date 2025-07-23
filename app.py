import streamlit as st
import pandas as pd
import numpy as np

# ─── Page Setup ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="B200 Landing Distance Calculator", layout="centered")
st.title("🛬 B200 King Air Landing Distance Estimator")

# ─── Step 1: User Inputs ────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    press_alt = st.slider("Pressure Altitude (ft)",   0, 10000, 2000, 250)
    oat       = st.slider("Outside Air Temperature (°C)", -5, 45, 15, 1)
with col2:
    weight = st.slider("Landing Weight (lb)", 9000, 12500, 11500, 100)
    wind   = st.slider("Wind Speed (kt)",     -20, 30, 0, 1,
                       help="Negative = tailwind, Positive = headwind")

# ─── Step 2: Table 1 – Pressure Altitude × OAT ──────────────────────────────
raw1 = pd.read_csv("pressureheight_oat.csv", skiprows=[0])
raw1 = raw1.rename(columns={raw1.columns[0]: "dummy", raw1.columns[1]: "PressAlt"})
tbl1 = raw1.drop(columns=["dummy"]).set_index("PressAlt")
tbl1.columns = tbl1.columns.astype(int)

def lookup_tbl1(df, pa, t):
    idx = max(i for i in df.index if i <= pa)
    hdr = max(h for h in df.columns if h <= t)
    return df.loc[idx, hdr]

baseline = lookup_tbl1(tbl1, press_alt, oat)
st.markdown("### Step 1: Baseline Distance")
st.success(f"Baseline landing distance: **{baseline:.0f} ft**")

# ─── Step 3: Table 2 – Weight Adjustment ────────────────────────────────────
raw2    = pd.read_csv("weightadjustment.csv", header=0)
wt_cols = [int(str(w).strip()) for w in raw2.columns]
df2     = raw2.astype(float)
df2.columns = wt_cols

def lookup_tbl2(df, baseline, w):
    ref12500 = df[12500]
    valid    = ref12500[ref12500 <= baseline]
    row      = valid.index.max() if not valid.empty else 0
    base12500 = ref12500.iloc[row]
    abs_w     = df.at[row, w]
    delta     = abs_w - base12500
    return baseline + delta

weight_adj = lookup_tbl2(df2, baseline, weight)
st.markdown("### Step 2: Weight Adjustment")
st.success(f"Weight-adjusted distance: **{weight_adj:.0f} ft**")

# ─── Step 4: Table 3 – Wind Adjustment ──────────────────────────────────────
raw3 = pd.read_csv("wind adjustment.csv", header=None)
wind_cols = [int(str(w).strip()) for w in raw3.iloc[0]]
df3 = raw3.iloc[1:].reset_index(drop=True).astype(float)
df3.columns = wind_cols

def lookup_tbl3(df, refd, ws):
    sorted_df = df.sort_values(by=0).reset_index(drop=True)
    ref0 = sorted_df[0]
    valid = ref0[ref0 <= refd]
    row = valid.index.max() if not valid.empty else 0
    base0 = ref0.iloc[row]
    abs_ws = sorted_df.at[row, ws]
    delta = abs_ws - base0
    return delta

delta_wind = lookup_tbl3(df3, weight_adj, wind)
wind_adj   = weight_adj + delta_wind
st.markdown("### Step 3: Wind Adjustment")
st.success(f"After wind adjustment: **{wind_adj:.0f} ft**")

# ─── Step 5: Table 4 – 50 ft Obstacle Correction ────────────────────────────
raw4 = pd.read_csv("50ft.csv", header=None)

# Check we have at least two columns
if raw4.shape[1] >= 2:
    df4 = raw4.iloc[:, :2].copy()
    df4.columns = [0, 50]
    df4 = df4.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

    def lookup_tbl4(df, refd):
        sorted_df = df.sort_values(by=0).reset_index(drop=True)
        ref_vals  = sorted_df[0].values
        obs_vals  = sorted_df[50].values
        return float(np.interp(
            refd,
            ref_vals,
            obs_vals,
            left=obs_vals[0],
            right=obs_vals[-1]
        ))

    obs50 = lookup_tbl4(df4, wind_adj)
    st.markdown("### Step 4: 50 ft Obstacle Correction")
    st.success(f"Final landing distance over 50 ft obstacle: **{obs50:.0f} ft**")

else:
    st.error("❌ 50ft.csv must have at least two columns: [ground roll, over-50ft]")
