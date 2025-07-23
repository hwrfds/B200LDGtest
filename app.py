import streamlit as st
import pandas as pd
import numpy as np

# ─── Page Setup ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="B200 Landing Distance Calculator", layout="centered")
st.title("🛬 B200 King Air Landing Distance Estimator")

# ─── Step 1: User Inputs ────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    press_alt = st.slider("Pressure Altitude (ft)", 0, 10000, 2000, 250)
    oat       = st.slider("Outside Air Temperature (°C)", -5, 45, 15, 1)
with col2:
    weight = st.slider("Landing Weight (lb)", 9000, 12500, 11500, 100)
    wind   = st.slider("Wind Speed (kt)", -20, 30, 0, 1,
                       help="Negative = tailwind, Positive = headwind")

# ─── Step 2: Table 1 – Pressure-Height × OAT ────────────────────────────────
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
st.write(f"Pressure Altitude: **{press_alt} ft**  \nOAT: **{oat} °C**")
st.success(f"Baseline landing distance: **{baseline:.0f} ft**")

# ─── Step 3: Table 2 – Weight Adjustment (interpolated) ────────────────────
raw2    = pd.read_csv("weightadjustment.csv", header=0)
wt_cols = [int(str(w).strip()) for w in raw2.columns]
df2     = raw2.astype(float)
df2.columns = wt_cols

def lookup_tbl2_interp(df, baseline, w):
    tbl       = df.sort_values(by=12500).reset_index(drop=True)
    ref12500  = tbl[12500].values
    weight_rs = tbl[w].values
    deltas    = weight_rs - ref12500
    delta_at_baseline = np.interp(
        baseline,
        ref12500,
        deltas,
        left=deltas[0],
        right=deltas[-1]
    )
    return baseline + float(delta_at_baseline)

weight_adj = lookup_tbl2_interp(df2, baseline, weight)
st.markdown("### Step 2: Weight Adjustment")
st.write(f"Selected Weight: **{weight} lb**")
st.success(f"Weight-adjusted distance: **{weight_adj:.0f} ft**")

# ─── Step 4: Table 3 – Wind Adjustment (interpolated) ───────────────────────
raw3      = pd.read_csv("wind adjustment.csv", header=None)
wind_cols = [int(str(w).strip()) for w in raw3.iloc[0]]
df3       = raw3.iloc[1:].reset_index(drop=True)
# convert every column to numeric, coerce errors to NaN
df3 = df3.apply(pd.to_numeric, errors='coerce')
df3.columns = wind_cols

def lookup_tbl3_interp(df, refd, ws):
    tbl        = df.sort_values(by=0).reset_index(drop=True)
    ref_rolls  = tbl[0].values
    wind_rolls = tbl[ws].values
    deltas     = wind_rolls - ref_rolls
    delta_at_r = np.interp(
        refd,
        ref_rolls,
        deltas,
        left=deltas[0],
        right=deltas[-1]
    )
    return float(delta_at_r)

delta_wind = lookup_tbl3_interp(df3, weight_adj, wind)
wind_adj   = weight_adj + delta_wind
st.markdown("### Step 3: Wind Adjustment")
st.write(f"Wind: **{wind:+.0f} kt** → Δ: **{delta_wind:.0f} ft**")
st.success(f"After wind adjustment: **{wind_adj:.0f} ft**")

# ─── Step 5: Table 4 – 50 ft Obstacle Correction (interpolated) ────────────
raw4 = pd.read_csv("50ft.csv", header=None)

# parse header row robustly into ints
hdr = raw4.iloc[0].tolist()
obs_cols = []
for cell in hdr:
    try:
        val = float(cell)
        obs_cols.append(int(val))
    except:
        # skip non-numeric headers
        continue

# grab the data rows, convert to numeric with coercion
data_rows = raw4.iloc[1:].reset_index(drop=True)
df4 = data_rows.apply(pd.to_numeric, errors='coerce')
df4.columns = obs_cols

def lookup_tbl4_interp(df, refd, obs_h=50):
    tbl       = df.sort_values(by=0).reset_index(drop=True)
    ref_rolls = tbl[0].values
    obs_rolls = tbl[obs_h].values
    return float(np.interp(
        refd,
        ref_rolls,
        obs_rolls,
        left=obs_rolls[0],
        right=obs_rolls[-1]
    ))

obs50 = lookup_tbl4_interp(df4, wind_adj, obs_h=50)
st.markdown("### Step 4: 50 ft Obstacle Correction")
st.success(f"Final landing distance over 50 ft obstacle: **{obs50:.0f} ft**")
