import streamlit as st
import pandas as pd
import numpy as np

# ─── Page Setup ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="RFDS QLD B200 Landing Distance Calculator", layout="centered")
st.title("🛬 RFDS QLD B200 King Air Landing Distance Calculator - NOT FOR OPERATIONAL USE")

# ─── Step 1: User Inputs ────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    press_alt = st.slider("Pressure Altitude (ft)",   0, 10000, 0, 200)
    oat       = st.slider("Outside Air Temperature (°C)", -5, 45, 15, 1)
with col2:
    weight = st.slider("Landing Weight (lb)", 9000, 12500, 11500, 100)
    wind   = st.slider("Wind Speed (kt)",     -10,    30,    0,   1,
                       help="Negative = tailwind, Positive = headwind")

# ─── Step 2: Table 1 – Pressure Altitude × OAT (Bilinear Interpolation) ───
raw1 = pd.read_csv("pressureheight_oat.csv", skiprows=[0])
raw1 = raw1.rename(columns={raw1.columns[0]: "dummy", raw1.columns[1]: "PressAlt"})
tbl1 = raw1.drop(columns=["dummy"]).set_index("PressAlt")
tbl1.columns = tbl1.columns.astype(int)

def lookup_tbl1_bilinear(df, pa, t):
    pas = np.array(sorted(df.index))
    oats = np.array(sorted(df.columns))
    pa  = np.clip(pa, pas[0], pas[-1])
    t   = np.clip(t,  oats[0], oats[-1])
    x1 = pas[pas <= pa].max()
    x2 = pas[pas >= pa].min()
    y1 = oats[oats <= t].max()
    y2 = oats[oats >= t].min()
    Q11 = df.at[x1, y1]; Q21 = df.at[x2, y1]
    Q12 = df.at[x1, y2]; Q22 = df.at[x2, y2]
    if x1 == x2 and y1 == y2:
        return Q11
    if x1 == x2:
        return Q11 + (Q12 - Q11) * (t - y1) / (y2 - y1)
    if y1 == y2:
        return Q11 + (Q21 - Q11) * (pa - x1) / (x2 - x1)
    denom = (x2 - x1) * (y2 - y1)
    fxy1 = Q11 * (x2 - pa) + Q21 * (pa - x1)
    fxy2 = Q12 * (x2 - pa) + Q22 * (pa - x1)
    return (fxy1 * (y2 - t) + fxy2 * (t - y1)) / denom

baseline = lookup_tbl1_bilinear(tbl1, press_alt, oat)
st.markdown("### Step 1: Baseline Distance")
st.success(f"Baseline landing distance: **{baseline:.0f} ft**")

# ─── Step 3: Table 2 – Weight Adjustment (1D Interpolation) ────────────────
raw2    = pd.read_csv("weightadjustment.csv", header=0)
wt_cols = [int(w) for w in raw2.columns]
df2     = raw2.astype(float)
df2.columns = wt_cols

def lookup_tbl2_interp(df, baseline, w, ref_weight=12500, _debug=False, _st=None):
    """
    Nearest-columns 2D interpolation on ABSOLUTE values (preferred).
    Returns the absolute weight-adjusted distance.
    """
    import numpy as np
    import pandas as pd

    tbl = df.copy()
    tbl.columns = [int(c) for c in tbl.columns]
    if ref_weight not in tbl.columns:
        raise ValueError(f"ref_weight {ref_weight} not found in columns")
    tbl = tbl.sort_values(by=ref_weight).reset_index(drop=True).astype(float)

    # X-axis tied to reference column (e.g., 12,500 lb)
    x_ref = tbl[ref_weight].values

    # Find nearest lower/upper weight columns
    weights = np.array(sorted(int(c) for c in tbl.columns))
    idx = int(np.searchsorted(weights, w, side="left"))
    if idx == 0:
        w1 = w2 = int(weights[0])
    elif idx >= len(weights):
        w1 = w2 = int(weights[-1])
    else:
        lower = int(weights[idx-1]); upper = int(weights[idx])
        w1, w2 = (upper, upper) if upper == w else (lower, upper)

    # Interpolate ABSOLUTE values in each bounding column at this baseline
    y1 = np.interp(baseline, x_ref, tbl[w1].values,
                   left=tbl[w1].values[0], right=tbl[w1].values[-1])
    y2 = np.interp(baseline, x_ref, tbl[w2].values,
                   left=tbl[w2].values[0], right=tbl[w2].values[-1])

    # Horizontal blend by proximity in weight
    if w1 == w2:
        y = y1; alpha = None
    else:
        alpha = (w - w1) / (w2 - w1)
        y = (1 - alpha) * y1 + alpha * y2

   # if _debug and (_st is not None):
    #    _st.caption(f"[WeightAdj ABS] weights={list(weights)} | w={w} → using {w1} & {w2} | alpha={alpha}")
     #   _st.caption(f"[WeightAdj ABS] baseline={baseline:.2f} | y1={y1:.2f} | y2={y2:.2f} | result={y:.2f}")

    return float(y)



weight_adj = lookup_tbl2_interp(df2, baseline, weight, _debug=True, _st=st)
st.markdown("### Step 2: Weight Adjustment")
st.success(f"Weight-adjusted distance: **{weight_adj:.0f} ft**")


# ─── Step 4: Table 3 – Wind Adjustment (1D Interpolation) ─────────────────
raw3      = pd.read_csv("wind adjustment.csv", header=None)
wind_cols = [int(w) for w in raw3.iloc[0]]
df3       = raw3.iloc[1:].reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
df3.columns = wind_cols

def lookup_tbl3_interp(df, refd, ws):
    tbl        = df.sort_values(by=0).reset_index(drop=True)
    ref_rolls  = tbl[0].values
    wind_rolls = tbl[ws].values
    deltas     = wind_rolls - ref_rolls
    delta_wind = np.interp(refd,
                           ref_rolls,
                           deltas,
                           left=deltas[0],
                           right=deltas[-1])
    return float(delta_wind)

delta_wind = lookup_tbl3_interp(df3, weight_adj, wind)
wind_adj   = weight_adj + delta_wind
st.markdown("### Step 3: Wind Adjustment")
st.success(f"After wind adjustment: **{wind_adj:.0f} ft**")

# ─── Step 5: Table 4 – 50 ft Obstacle Correction (1D Interpolation) ────────
raw4 = pd.read_csv("50ft.csv", header=None)
df4  = raw4.iloc[:, :2].copy()
df4.columns = [0, 50]
df4 = df4.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

def lookup_tbl4_interp(df, refd, h=50, ref_col=0, _debug=False, _st=None):
    """
    2D ABSOLUTE interpolation for the 50 ft obstacle table (or any height h):
      - x-axis: reference distances in column `ref_col` (e.g., 0 ft obstacle).
      - y-axis: absolute distances in the two nearest obstacle-height columns around `h`.
      - returns the absolute distance at obstacle height h.
    """
    import numpy as np
    import pandas as pd

    tbl = df.copy().astype(float)

    # Collect obstacle height columns (all except ref_col)
    all_cols = [c for c in tbl.columns]
    # Convert to numeric if needed
    try:
        all_cols_num = [int(c) for c in all_cols]
    except Exception:
        all_cols_num = []
        for c in all_cols:
            try: all_cols_num.append(int(c))
            except: all_cols_num.append(c)

    # Map back to original labels
    colmap = {int(c): c for c in all_cols if str(c).isdigit()}
    if ref_col not in colmap:
        # ref_col may already be the exact label
        ref_label = ref_col
    else:
        ref_label = colmap[ref_col]

    # Sort by the reference column
    tbl = tbl.sort_values(by=ref_label).reset_index(drop=True)

    x_ref = tbl[ref_label].values

    # Candidate obstacle columns (numeric only, excluding ref_col)
    obs_heights = sorted([k for k in colmap.keys() if k != ref_col])

    # Find nearest lower/upper heights around h
    import bisect
    idx = bisect.bisect_left(obs_heights, h)
    if idx == 0:
        h1 = h2 = obs_heights[0]
    elif idx >= len(obs_heights):
        h1 = h2 = obs_heights[-1]
    else:
        lower = obs_heights[idx-1]; upper = obs_heights[idx]
        h1, h2 = (upper, upper) if upper == h else (lower, upper)

    # Interpolate ABS values in each obstacle column at this refd
    y1 = np.interp(refd, x_ref, tbl[colmap[h1]].values,
                   left=tbl[colmap[h1]].values[0], right=tbl[colmap[h1]].values[-1])
    y2 = np.interp(refd, x_ref, tbl[colmap[h2]].values,
                   left=tbl[colmap[h2]].values[0], right=tbl[colmap[h2]].values[-1])

    # Horizontal blend by obstacle height
    if h1 == h2:
        y = y1; alpha = None
    else:
        alpha = (h - h1) / (h2 - h1)
        y = (1 - alpha) * y1 + alpha * y2

  #  if _debug and (_st is not None):
   #     _st.caption(f"[Obs ABS] heights={obs_heights} | h={h} → using {h1} & {h2} | alpha={alpha}")
    #    _st.caption(f"[Obs ABS] refd={refd:.2f} | y1={y1:.2f} | y2={y2:.2f} | result={y:.2f}")

    return float(y)



obs50 = lookup_tbl4_interp(df4, wind_adj, h=50, ref_col=0, _debug=True, _st=st)
st.markdown("### Step 4: 50 ft Obstacle Correction")
st.success(f"Final landing distance over 50 ft obstacle: **{obs50:.0f} ft**")

# ─── Additional Output: Distance in Meters ─────────────────────────────────
obs50_m = obs50 * 0.3048
st.markdown("### Final Landing Distance in Meters")
st.success(f"{obs50_m:.1f} m")

# ─── Step 6: Apply a Factor ───────────────────────────────────
factor_options = {
    "Standard Factor Dry (1.43)": 1.43,
    "Standard Factor Wet (1.65)": 1.65,
    "Approved Factor Dry (1.20)": 1.20,
    "Approved Factor Wet (1.38)": 1.38,
}

# show dropdown and grab the numeric value
factor_label = st.selectbox(
    "Select Landing Distance Factor",
    list(factor_options.keys())
)
factor = factor_options[factor_label]

# apply factor to the raw over-50 ft distance
factored_ft = obs50 * factor
factored_m  = factored_ft * 0.3048

# display results side-by-side
st.markdown("### Factored Landing Distance")
col1, col2 = st.columns(2)
col1.success(f"{factored_ft:.0f} ft")
col2.success(f"{factored_m:.1f} m")



# ─── Step X: Runway Slope Input & Adjustment (negative = downslope) ─────────
slope_deg = st.number_input(
    "Runway Slope (%)",
    min_value=-5.0,
    max_value= 0.0,
    value= 0.0,
    step= 0.1,
    help="Negative = downslope (increases distance), Positive = upslope (no effect)"
)

# For negative slope values, apply 20% extra distance per 1% downslope
slope_factor = 1.0 + max(-slope_deg, 0.0) * 0.20

# Apply slope factor to the already factored landing distance (ft)
sloped_ft = factored_ft * slope_factor
sloped_m  = sloped_ft * 0.3048

st.markdown("### Slope Adjustment")
col1, col2 = st.columns(2)
col1.write(f"**Slope:** {slope_deg:+.1f}%")
col2.write(f"**Slope Factor:** ×{slope_factor:.2f}")

col3, col4 = st.columns(2)
col3.success(f"Distance w/ Slope: **{sloped_ft:.0f} ft**")
col4.success(f"Distance w/ Slope: **{sloped_m:.1f} m**")

# ─── Step Y: Landing Distance Available & Go/No-Go ─────────────────────────
# Input available runway length in metres
avail_m = st.number_input(
    "Landing Distance Available (m)",
    min_value=0.0,
    value=1150.0,
    step=5.0,
    help="Enter the runway length available in metres"
)

# Convert to feet
avail_ft = avail_m / 0.3048

# Display the available distance
st.markdown("### Available Runway Length")
c1, c2 = st.columns(2)
c1.write(f"**{avail_m:.0f} m**")
c2.write(f"**{avail_ft:.0f} ft**")

# Determine if tailwind exists (positive wind value)
has_tailwind = wind < 0

# Check if the 1.20 factor is selected
using_1_2_factor = factor_label == "Approved Factor Dry (1.20)"

# Go/No-Go Decision Logic
st.markdown("### Go/No-Go Decision")

if using_1_2_factor and has_tailwind:
    st.error("❌ Landing not permitted: No tailwind component permitted with 1.2 Factoring")
elif avail_ft >= sloped_ft:
    st.success("✅ Enough runway available for landing")
else:
    st.error("❌ Insufficient runway available for landing")


st.markdown("### Data extracted from B200-601-228 HFG Perfomance Landing Distance w Propeller Reversing - Flap 100%")


st.markdown("Created by H Watson and R Thomas")
