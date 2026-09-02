# %%
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
from scipy.optimize import curve_fit

pd.set_option('display.max_columns', None)
# %%
KPI_PATH = "../data/fcst/product_type_kpi_month.csv"
PRODUCT_TYPES_PATH = "../data/fcst/product_types.csv"
MONTH_INDEX_START = pd.Timestamp("2026-01-01")


def load_ad_sales_data(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.lower()
    frame["product_range"] = frame["product_range"].fillna("").str.strip()
    frame = frame[frame["product_range"] != ""].copy()
    frame["ds"] = pd.to_datetime(frame["month_year"], format="%m-%Y")
    frame["ad_sales"] = frame["ad_sales"].clip(lower=0)
    frame["ad_spend"] = frame["ad_spend"].clip(lower=0)
    frame = frame[frame['ds']<'2026-04-01'][['ds', 'product_range', 'ad_sales', 'ad_spend']]
    return frame.sort_values(["product_range", "ds"])


def build_product_type_features(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["Parent Category"] = frame["Parent Category"].fillna("")
    frame["Kitchen"] = frame["Parent Category"].str.contains("Kitchen").astype(int)
    frame["Health"] = frame["Parent Category"].str.contains("Health").astype(int)
    frame["Industrial"] = frame["Parent Category"].str.contains("Industrial").astype(int)
    frame["Tools"] = frame["Parent Category"].str.contains("Tools").astype(int)
    frame["Automotive"] = frame["Parent Category"].str.contains("Automotive").astype(int)
    frame["Pet"] = frame["Parent Category"].str.contains("Pet").astype(int)
    feature_df = frame.groupby("Product Type", as_index=False).agg(
        {
            "Product ID": "count",
            "Price": "mean",
            "Kitchen": "sum",
            "Health": "sum",
            "Industrial": "sum",
            "Tools": "sum",
            "Automotive": "sum",
            "Pet": "sum",
            "Category": "nunique",
            "Size": "nunique",
            "Colour": "nunique",
        }
    )
    return feature_df.rename(
        columns={
            "Product Type": "product_range",
            "Product ID": "product_id_count",
            "Price": "price_avg",
            "Category": "category_nunique",
            "Size": "size_nunique",
            "Colour": "colour_nunique",
            "Kitchen": "kitchen_count",
            "Health": "health_count",
            "Industrial": "industrial_count",
            "Tools": "tools_count",
            "Automotive": "automotive_count",
            "Pet": "pet_count",
        }
    )


def merge_static_features(ad_df: pd.DataFrame, static_df: pd.DataFrame) -> pd.DataFrame:
    merged = ad_df.merge(static_df, on="product_range", how="left")
    static_cols = [col for col in static_df.columns if col != "product_range"]
    for column in static_cols:
        merged[column] = merged[column].fillna(0)
    return merged


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    lagged = frame.copy()
    grouped = lagged.groupby("product_range")
    lagged["ad_sales_lag_1"] = grouped["ad_sales"].shift(1)
    lagged["ad_sales_lag_2"] = grouped["ad_sales"].shift(2)
    lagged["ad_sales_lag_3"] = grouped["ad_sales"].shift(3)
    lagged["ad_spend_lag_1"] = grouped["ad_spend"].shift(1)
    lagged["ad_spend_lag_2"] = grouped["ad_spend"].shift(2)
    lagged["ad_spend_lag_3"] = grouped["ad_spend"].shift(3)
    lagged["ad_spend_lag_1_rolling3"] = grouped["ad_spend"].shift(1).rolling(3).mean()
    lagged["month"] = lagged["ds"].dt.month
    lagged["quarter"] = lagged["ds"].dt.quarter
    return lagged

# def hill_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
#     """Hill function (sigmoidal) for curve fitting."""
#     return a * x**b / (c**b + x**b)

def hill(x: np.ndarray | float, ec: float, slope: float) -> np.ndarray:
    """
    Hill(x; ec, slope) = 1 / (1 + (x/ec)^(-slope))
    """
    x_arr = np.asarray(x, dtype=float)
    ec_safe = max(float(ec), 1e-9)
    x_safe = np.clip(x_arr, 1e-12, None)
    return 1.0 / (1.0 + (x_safe / ec_safe) ** (-float(slope)))

def hill_scaled(x: np.ndarray | float, ymax: float, ec: float, slope: float) -> np.ndarray:
    return float(ymax) * hill(x, ec, slope)

def hill_sat(x, ymax, ec, slope):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 1e-9, None)      # avoid 0^negative issues
    ec = max(float(ec), 1e-9)
    slope = max(float(slope), 1e-6)
    return ymax / (1.0 + (x / ec) ** (-slope))
# %%
# %%
pos_df = pd.read_parquet('../data/fcst/pos_df.parquet')
future_df = pd.read_parquet('../data/fcst/future_predictions.parquet')

plandf = pd.read_csv('../data/fcst/bissell_plans.csv')
for col in plandf.columns[1:]:
    plandf[col] = plandf[col].replace(r'[\$,]', '', regex=True).astype(float)


ad_df = load_ad_sales_data(KPI_PATH)
product_type_features_df = build_product_type_features(PRODUCT_TYPES_PATH)
adp_df = merge_static_features(ad_df, product_type_features_df)
feature_df = add_lag_features(adp_df)

feature_cols = [
    "product_range",
    "ad_spend",
    "ad_sales_lag_1",
    "ad_sales_lag_2",
    "ad_sales_lag_3",
    "ad_spend_lag_1",
    "ad_spend_lag_2",
    "ad_spend_lag_3",
    "ad_spend_lag_1_rolling3",
    "month",
    "quarter",
    "product_id_count",
    "price_avg",
    "kitchen_count",
    "health_count",
    "industrial_count",
    "tools_count",
    "automotive_count",
    "pet_count",
    "category_nunique",
    "size_nunique",
    "colour_nunique",
]
categorical_cols = ["product_range"]
# %%
feature_df[feature_df['ad_spend']==0].sample(frac=0.1)
# %%
zero_df = feature_df[feature_df['ad_spend']==0].sample(frac=0.1)
feature_df = feature_df[(feature_df['ad_spend']>0)].copy()
feature_df = pd.concat([feature_df, zero_df])

future_df.loc[future_df['month_year']=='06-2026', 'ds'] = pd.Timestamp('2026-07-01')
future_df.loc[future_df['month_year']=='07-2026', 'ds'] = pd.Timestamp('2026-06-01')


train_df = feature_df[feature_df['ds']<'2026-03-01'].dropna(subset=feature_cols).copy()
test_df = feature_df[feature_df['ds']>='2026-03-01'].dropna(subset=feature_cols).copy()
train_df["target_log"] = np.log1p(train_df["ad_sales"])
train_df["product_range"] = train_df["product_range"].astype("category")
test_df["product_range"] = test_df["product_range"].astype("category")
# %%
model = LGBMRegressor(
    n_estimators=900,
    learning_rate=0.03,
    num_leaves=40,
    min_child_samples=10,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
)
model.fit(
    train_df[feature_cols],
    train_df["target_log"],
    categorical_feature=categorical_cols,
)

test_df['pred_log'] = model.predict(test_df[feature_cols])
test_df['pred_ad_sales'] = np.expm1(test_df['pred_log'])
# px.scatter(test_df, x='ad_sales', y='pred_ad_sales')
mape(test_df['ad_sales'], test_df['pred_ad_sales'])
# %%
train_df = feature_df.dropna(subset=feature_cols).copy()
train_df["target_log"] = np.log1p(train_df["ad_sales"])
for column in categorical_cols:
    train_df[column] = train_df[column].astype("category")
model.fit(
    train_df[feature_cols],
    train_df["target_log"],
    categorical_feature=categorical_cols,
)
# %%
month = 4
# prev_month = ad_df[ad_df['ds']==f'2026-{month-1}-01'].copy()
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 630997.9726964167, 1: 730916.2681919568, 2: 266239.3019594477, 3: 405978.11886186595, 4: 60002.98561356372, 5: 68633.803285152, 6: 101458.90622953199}, 'ad_sales': {0: 6076872.2005868545, 1: 3383345.337969808, 2: 2426520.789387737, 3: 2366990.0787871964, 4: 842564.5508048449, 5: 263491.00650160335, 6: 470946.91946802585}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
new_ad_df = pd.concat([ad_df, prev_month])

month = 5
# prev_month = new_ad_df[new_ad_df['ds']==f'2026-{month-1}-01'].copy()
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 132841.6784624035, 1: 17541.990436606964, 2: 266239.3019594477, 3: 844434.4872326812, 4: 666699.840150708, 5: 82360.56394218239, 6: 3652.520624263151}, 'ad_sales': {0: 3017292.265501146, 1: 491959.613729535, 2: 2493833.825687014, 3: 2778309.718883705, 4: 3023247.2234902596, 5: 277087.4294837087, 6: 112463.39446206397}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
new_ad_df = pd.concat([new_ad_df, prev_month])

month = 6
# prev_month = new_ad_df[new_ad_df['ds']==f'2026-{month-1}-01'].copy()
# prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 1660520.9807800439, 1: 324526.82307722885, 2: 1663995.637246548, 3: 1623912.4754474638, 4: 800039.8081808495, 5: 98832.67673061886, 6: 60875.34373771919}, 'ad_sales': {0: 9344682.274191564, 1: 3679570.9008986587, 2: 4314017.176782616, 3: 3902615.6849041614, 4: 3028630.27892828, 5: 167212.43881577358, 6: 635359.510419369}})
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 149446.88827020396, 1: 184190.8995843731, 2: 1663995.637246548, 3: 1623912.4754474638, 4: 8000.398081808496, 5: 988.3267673061886, 6: 10957.561872789454}, 'ad_sales': {0: 4359162.385514504, 1: 2640792.37893786, 2: 4314017.176782616, 3: 3902615.6849041614, 4: 87534.2080615518, 5: 29755.034167307276, 6: 136799.89459337754}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
new_ad_df = pd.concat([new_ad_df, prev_month])

month = 7
# prev_month = new_ad_df[new_ad_df['ds']==f'2026-{month-1}-01'].copy()
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 199262.51769360527, 1: 140335.9234928557, 2: 119807.68588175144, 3: 545634.5917503479, 4: 8000.398081808496, 5: 0.0, 6: 121750.68747543838}, 'ad_sales': {0: 6309955.205508002, 1: 2375292.952521454, 2: 1693526.4411476485, 3: 3393521.3812513975, 4: 223860.8614543638, 5: 1.5001591835520923e-07, 6: 537812.5080064018}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
new_ad_df = pd.concat([new_ad_df, prev_month])

month = 8
# prev_month = new_ad_df[new_ad_df['ds']==f'2026-{month-1}-01'].copy()
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 83026.04903900219, 1: 78938.95696473133, 2: 119807.68588175144, 3: 97434.74852684783, 4: 8000.398081808496, 5: 988.3267673061886, 6: 8766.049498231561}, 'ad_sales': {0: 2510984.282180078, 1: 1949514.4865875267, 2: 1843734.3580056394, 3: 852587.3863432906, 4: 91080.41756477694, 5: 6390.317395409097, 6: 258995.95185590946}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
new_ad_df = pd.concat([new_ad_df, prev_month])

month = 9
# prev_month = new_ad_df[new_ad_df['ds']==f'2026-{month-1}-01'].copy()
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 83026.04903900219, 1: 43854.97609151741, 2: 119807.68588175144, 3: 253330.34616980437, 4: 8000.398081808496, 5: 988.3267673061886, 6: 2922.0164994105207}, 'ad_sales': {0: 3520007.133074691, 1: 1189963.721296074, 2: 2102560.9129915796, 3: 2106106.2674132558, 4: 137781.7485854679, 5: 245759.2853793762, 6: 47856.46911541038}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
new_ad_df = pd.concat([new_ad_df, prev_month])

month = 10
# prev_month = new_ad_df[new_ad_df['ds']==f'2026-{month-1}-01'].copy()
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 1660520.9807800439, 1: 149106.9187111592, 2: 1996794.7646958574, 3: 1948694.9705369566, 4: 8000.398081808496, 5: 988.3267673061886, 6: 146100.82497052604}, 'ad_sales': {0: 7727440.8540801285, 1: 1831616.251834248, 2: 5408718.317737302, 3: 3323496.6992286914, 4: 103854.2398630553, 5: 220124.52129406523, 6: 810387.4425619927}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
new_ad_df = pd.concat([new_ad_df, prev_month])

month = 11
# prev_month = new_ad_df[new_ad_df['ds']==f'2026-{month-1}-01'].copy()
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 1155722.6026229104, 1: 52625.97130982089, 2: 2000000.0, 3: 20000.0, 4: 8000.398081808496, 5: 988.3267673061886, 6: 0.0}, 'ad_sales': {0: 5291128.784351846, 1: 946003.8043358118, 2: 3631662.4058167413, 3: 198212.03676245367, 4: 120462.79256415347, 5: 271628.2675972582, 6: 1.5930977038135e-08}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
new_ad_df = pd.concat([new_ad_df, prev_month])

month = 12
# prev_month = new_ad_df[new_ad_df['ds']==f'2026-{month-1}-01'].copy()
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 876755.0778518631, 1: 368381.7991687462, 2: 2000000.0, 3: 2000000.0, 4: 8000.398081808496, 5: 988.3267673061886, 6: 175320.98996463124}, 'ad_sales': {0: 6832274.196038563, 1: 2419789.7031691913, 2: 4907246.723110476, 3: 3873240.4345592195, 4: 120197.22209166695, 5: 277927.7942423209, 6: 65632.66171743645}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
new_ad_df = pd.concat([new_ad_df, prev_month])

# %%
adp_df = merge_static_features(new_ad_df, product_type_features_df)
feature_df = add_lag_features(adp_df)
pred_df = feature_df[feature_df['ds']==f'2026-{month}-01']

max_spend = feature_df.groupby('product_range')['ad_spend'].max()*1.2
max_spend = max_spend.map(lambda x: 10e3 if x<1e3 else x)
max_spend = max_spend.map(lambda x:  2e6 if x>2e6 else x)
max_spend = max_spend.map(lambda x:  np.linspace(0, x, 101))
spend_grid = pd.DataFrame(max_spend).explode('ad_spend').reset_index()
spend_grid['ad_spend'] = spend_grid['ad_spend'].astype(float)
pred_df = pd.merge(pred_df.drop('ad_spend', axis=1), spend_grid, on='product_range', how='left')

pred_df['product_range'] = pred_df['product_range'].astype("category")
pred_df['pred_log'] = model.predict(pred_df[feature_cols])
pred_df['pred_ad_sales'] = np.expm1(pred_df['pred_log'])
pred_df.loc[pred_df['ad_spend']==0, 'pred_ad_sales'] = 0
pred_df

# %%
predlist = []
prods = ['PDC', 'UDC', 'UVAC', 'Stick', 'Steam', 'Sanitaire', 'Formula']
for prod in prods: 
    df = pred_df[pred_df['product_range'] == prod].copy()

    xdata = df["ad_spend"].values
    ydata = df["pred_ad_sales"].values

    x = df["ad_spend"].to_numpy(float)
    y = df["pred_ad_sales"].to_numpy(float)
    
    ord_idx = np.argsort(x)
    x, y = x[ord_idx], y[ord_idx]
    p0 = [
        max(y.max(), 1.0),                             # ymax
        np.median(x[x > 0]) if np.any(x > 0) else 1,  # ec
        0.8                                           # slope
    ]
    lower = [0.0, 1e-6, 0.2]
    upper = [1.5 * max(y.max(), 1.0), max(x.max(), 1.0) * 2.0, 1.0]
    popt, _ = curve_fit(
        hill_sat,
        x, y,
        p0=p0,
        bounds=(lower, upper),
        maxfev=50000
    )
    df["ad_sales"] = hill_sat(df["ad_spend"].to_numpy(float), *popt)

    pdf = pos_df[pos_df['Product Type'] == prod].astype({'month': 'datetime64[ns]'})
    fdf = future_df[future_df['product_range'] == prod].astype({'ds': 'datetime64[ns]'})
    pdf = pdf.set_index('month')
    fdf = fdf.set_index('ds')
    pdc_df = pdf.join(fdf, on='month')

    pdc_df['exp_ad_sales'] = pdc_df['pos'] - pdc_df['organic_sales_forecast']
    ad_sales_goal = int(pdc_df.loc[f'2026-{month}-01', 'exp_ad_sales'])

    a = df[['product_range', 'ad_spend', 'ad_sales']]
    goal_spend = a[a['ad_sales']<ad_sales_goal]

    if ad_sales_goal<0:
        goal_spend = a.iloc[1] if goal_spend.empty else goal_spend.iloc[-1]
        predlist.append(goal_spend)
    elif goal_spend.iloc[-1]['ad_sales']<ad_sales_goal:
        goal_spend = goal_spend.iloc[-1]
        predlist.append(goal_spend)
    else:
        goal_spend = goal_spend[goal_spend['ad_sales']<ad_sales_goal].iloc[-1]
        predlist.append(goal_spend)

# %%
px.line(df, x='ad_spend', y=['pred_ad_sales', 'ad_sales'])
# a = feature_df[feature_df['product_range']==prod]
# px.scatter(a, x='ad_spend', y='ad_sales')
# %%
# df = feature_df[feature_df['product_range']=='PDC']
# px.line(df, x='ds', y=['ad_spend', 'ad_sales'])
# %%
propdf = pd.DataFrame(predlist)
propdf
# %%
df = pd.merge(propdf, plandf, left_on='product_range', right_on='Category', how='right')
px.bar(df, x='product_range', y=['December', 'ad_spend'], barmode='group')
# %%
# a = ad_df[ad_df['product_range']=='PDC']
# fig = px.line(a, x='ds', y=['ad_spend', 'ad_sales'])
# fig.show()
# %%
print(propdf.reset_index(drop=True).to_dict())
# %%
pos_df[pos_df['Product Type']=='PDC']
# %%
months = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
# %%
future_df['month_name'] = future_df['ds'].dt.strftime('%B')
df = future_df.pivot(index='product_range', columns='month_name', values='organic_sales_forecast')
df[months].astype(int).to_csv('../data/fcst/out_fcst.csv')
# %%
pos_df['month_name'] = pos_df['month'].astype('datetime64[ns]').dt.strftime('%B')
df = pos_df.pivot(index='Product Type', columns='month_name', values='pos')
df[months].astype(int).to_csv('../data/fcst/out_pos.csv')
# %%
plandf.set_index('Category')[months].astype(int).to_csv('../data/fcst/out_plans.csv')
# %%
planlist = []
month = 4
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 630997.9726964167, 1: 730916.2681919568, 2: 266239.3019594477, 3: 405978.11886186595, 4: 60002.98561356372, 5: 68633.803285152, 6: 101458.90622953199}, 'ad_sales': {0: 6076872.2005868545, 1: 3383345.337969808, 2: 2426520.789387737, 3: 2366990.0787871964, 4: 842564.5508048449, 5: 263491.00650160335, 6: 470946.91946802585}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
planlist += [prev_month]

month = 5
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 132841.6784624035, 1: 17541.990436606964, 2: 266239.3019594477, 3: 844434.4872326812, 4: 666699.840150708, 5: 82360.56394218239, 6: 3652.520624263151}, 'ad_sales': {0: 3017292.265501146, 1: 491959.613729535, 2: 2493833.825687014, 3: 2778309.718883705, 4: 3023247.2234902596, 5: 277087.4294837087, 6: 112463.39446206397}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
planlist += [prev_month]

month = 6
# prev_month = new_ad_df[new_ad_df['ds']==f'2026-{month-1}-01'].copy()
# prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 1660520.9807800439, 1: 324526.82307722885, 2: 1663995.637246548, 3: 1623912.4754474638, 4: 800039.8081808495, 5: 98832.67673061886, 6: 60875.34373771919}, 'ad_sales': {0: 9344682.274191564, 1: 3679570.9008986587, 2: 4314017.176782616, 3: 3902615.6849041614, 4: 3028630.27892828, 5: 167212.43881577358, 6: 635359.510419369}})
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 149446.88827020396, 1: 184190.8995843731, 2: 1663995.637246548, 3: 1623912.4754474638, 4: 8000.398081808496, 5: 988.3267673061886, 6: 10957.561872789454}, 'ad_sales': {0: 4359162.385514504, 1: 2640792.37893786, 2: 4314017.176782616, 3: 3902615.6849041614, 4: 87534.2080615518, 5: 29755.034167307276, 6: 136799.89459337754}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
planlist += [prev_month]

month = 7
# prev_month = new_ad_df[new_ad_df['ds']==f'2026-{month-1}-01'].copy()
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 199262.51769360527, 1: 140335.9234928557, 2: 119807.68588175144, 3: 545634.5917503479, 4: 8000.398081808496, 5: 0.0, 6: 121750.68747543838}, 'ad_sales': {0: 6309955.205508002, 1: 2375292.952521454, 2: 1693526.4411476485, 3: 3393521.3812513975, 4: 223860.8614543638, 5: 1.5001591835520923e-07, 6: 537812.5080064018}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
planlist += [prev_month]

month = 8
# prev_month = new_ad_df[new_ad_df['ds']==f'2026-{month-1}-01'].copy()
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 83026.04903900219, 1: 78938.95696473133, 2: 119807.68588175144, 3: 97434.74852684783, 4: 8000.398081808496, 5: 988.3267673061886, 6: 8766.049498231561}, 'ad_sales': {0: 2510984.282180078, 1: 1949514.4865875267, 2: 1843734.3580056394, 3: 852587.3863432906, 4: 91080.41756477694, 5: 6390.317395409097, 6: 258995.95185590946}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
planlist += [prev_month]

month = 9
# prev_month = new_ad_df[new_ad_df['ds']==f'2026-{month-1}-01'].copy()
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 83026.04903900219, 1: 43854.97609151741, 2: 119807.68588175144, 3: 253330.34616980437, 4: 8000.398081808496, 5: 988.3267673061886, 6: 2922.0164994105207}, 'ad_sales': {0: 3520007.133074691, 1: 1189963.721296074, 2: 2102560.9129915796, 3: 2106106.2674132558, 4: 137781.7485854679, 5: 245759.2853793762, 6: 47856.46911541038}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
planlist += [prev_month]

month = 10
# prev_month = new_ad_df[new_ad_df['ds']==f'2026-{month-1}-01'].copy()
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 1660520.9807800439, 1: 149106.9187111592, 2: 1996794.7646958574, 3: 1948694.9705369566, 4: 8000.398081808496, 5: 988.3267673061886, 6: 146100.82497052604}, 'ad_sales': {0: 7727440.8540801285, 1: 1831616.251834248, 2: 5408718.317737302, 3: 3323496.6992286914, 4: 103854.2398630553, 5: 220124.52129406523, 6: 810387.4425619927}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
planlist += [prev_month]

month = 11
# prev_month = new_ad_df[new_ad_df['ds']==f'2026-{month-1}-01'].copy()
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 1155722.6026229104, 1: 52625.97130982089, 2: 2000000.0, 3: 20000.0, 4: 8000.398081808496, 5: 988.3267673061886, 6: 0.0}, 'ad_sales': {0: 5291128.784351846, 1: 946003.8043358118, 2: 3631662.4058167413, 3: 198212.03676245367, 4: 120462.79256415347, 5: 271628.2675972582, 6: 1.5930977038135e-08}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
planlist += [prev_month]

month = 12
# prev_month = new_ad_df[new_ad_df['ds']==f'2026-{month-1}-01'].copy()
prev_month = pd.DataFrame({'product_range': {0: 'PDC', 1: 'UDC', 2: 'UVAC', 3: 'Stick', 4: 'Steam', 5: 'Sanitaire', 6: 'Formula'}, 'ad_spend': {0: 876755.0778518631, 1: 368381.7991687462, 2: 2000000.0, 3: 2000000.0, 4: 8000.398081808496, 5: 988.3267673061886, 6: 175320.98996463124}, 'ad_sales': {0: 6832274.196038563, 1: 2419789.7031691913, 2: 4907246.723110476, 3: 3873240.4345592195, 4: 120197.22209166695, 5: 277927.7942423209, 6: 65632.66171743645}})
prev_month['ds'] = pd.Timestamp(f'2026-{month}-01')
planlist += [prev_month]

prop_df = pd.concat(planlist)
prop_df
# %%
prop_df['month_name'] = prop_df['ds'].dt.strftime('%B')
df = prop_df.pivot(index='product_range', columns='month_name', values='ad_spend')
df[months].astype(int).to_csv('../data/fcst/out_prop.csv')

# %%
