# %%
from typing import Final

import pandas as pd
import plotly.express as px
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences

KPI_PATH: Final[str] = "../data/fcst/product_type_kpi_month.csv"
PRODUCT_TYPES_PATH: Final[str] = "../data/fcst/product_types.csv"
FORECAST_START: Final[pd.Timestamp] = pd.Timestamp("2026-04-01")
FORECAST_END: Final[pd.Timestamp] = pd.Timestamp("2026-12-01")
MIN_SERIES_LEN_FOR_YOY_DIFF: Final[int] = 13
TEST_HORIZON: Final[int] = 3


def build_product_type_features(product_types: pd.DataFrame) -> pd.DataFrame:
    product_types = product_types.copy()
    product_types["Parent Category"] = product_types["Parent Category"].fillna("")
    product_types["Kitchen"] = product_types["Parent Category"].str.contains("Kitchen").astype(int)
    product_types["Health"] = product_types["Parent Category"].str.contains("Health").astype(int)
    product_types["Industrial"] = (
        product_types["Parent Category"].str.contains("Industrial").astype(int)
    )
    product_types["Tools"] = product_types["Parent Category"].str.contains("Tools").astype(int)
    product_types["Automotive"] = (
        product_types["Parent Category"].str.contains("Automotive").astype(int)
    )
    product_types["Pet"] = product_types["Parent Category"].str.contains("Pet").astype(int)
    return product_types.groupby("Product Type", as_index=False).agg(
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


def prepare_training_frame(
    kpi_df: pd.DataFrame, product_features: pd.DataFrame
) -> pd.DataFrame:
    frame = kpi_df.copy()
    frame.columns = frame.columns.str.lower()
    frame["product_range"] = frame["product_range"].fillna("").str.strip()
    frame = frame[frame["product_range"] != ""].copy()
    frame["date"] = pd.to_datetime(frame["month_year"], format="%m-%Y")
    frame = frame.merge(
        product_features,
        left_on="product_range",
        right_on="Product Type",
        how="left",
    )
    frame["Price"] = frame["Price"].fillna(frame["Price"].median())
    return frame


def build_mlforecast_model() -> MLForecast:
    model = LGBMRegressor(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=8,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    return MLForecast(
        models=[model],
        freq="MS",
        lags=[1, 2, 3, 6, 12],
        date_features=["month", "quarter"],
        target_transforms=[Differences([12])],
    )


def forecast_with_fallback(
    model_input_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    static_features: list[str],
) -> pd.DataFrame:
    series_sizes = model_input_df.groupby("unique_id").size().rename("n_obs").reset_index()
    eligible_ids = series_sizes[
        series_sizes["n_obs"] >= MIN_SERIES_LEN_FOR_YOY_DIFF
    ]["unique_id"].tolist()
    short_ids = series_sizes[series_sizes["n_obs"] < MIN_SERIES_LEN_FOR_YOY_DIFF][
        "unique_id"
    ].tolist()

    fit_df_eligible = model_input_df[model_input_df["unique_id"].isin(eligible_ids)].copy()
    forecast_model = build_mlforecast_model()
    forecast_model.fit(
        fit_df_eligible,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=static_features,
    )

    horizon = len(pd.date_range(start=start_date, end=end_date, freq="MS"))
    future_predictions_model = forecast_model.predict(h=horizon).rename(
        columns={"LGBMRegressor": "organic_sales_forecast"}
    )
    future_predictions_model = future_predictions_model[
        (future_predictions_model["ds"] >= start_date)
        & (future_predictions_model["ds"] <= end_date)
    ].copy()

    future_months = pd.date_range(start=start_date, end=end_date, freq="MS")
    fallback_rows = []
    for product_range in short_ids:
        last_value = model_input_df.loc[
            model_input_df["unique_id"] == product_range, "y"
        ].iloc[-1]
        for month in future_months:
            fallback_rows.append(
                {
                    "unique_id": product_range,
                    "ds": month,
                    "organic_sales_forecast": float(last_value),
                }
            )
    future_predictions_fallback = pd.DataFrame(fallback_rows)
    predictions = pd.concat(
        [future_predictions_model, future_predictions_fallback], ignore_index=True
    )
    predictions["organic_sales_forecast"] = predictions["organic_sales_forecast"].clip(
        lower=0
    )
    predictions["month_year"] = predictions["ds"].dt.strftime("%m-%Y")
    return predictions.rename(columns={"unique_id": "product_range"})[
        ["month_year", "ds", "product_range", "organic_sales_forecast"]
    ].sort_values(["product_range", "ds"])

# %%
kpi_df = pd.read_csv(KPI_PATH)
product_types_df = pd.read_csv(PRODUCT_TYPES_PATH)

product_type_features_df = build_product_type_features(product_types_df)
training_df = prepare_training_frame(kpi_df, product_type_features_df)

static_features = [
    "Product ID",
    "Price",
    "Kitchen",
    "Health",
    "Industrial",
    "Pet",
    "Tools",
    "Automotive",
    "Category",
    "Size",
    "Colour",
    "PrimeDay"
]

series_df = training_df.rename(
    columns={"product_range": "unique_id", "date": "ds", "organic_sales": "y"}
)[["unique_id", "ds", "y", *static_features]].sort_values(["unique_id", "ds"])
# %%
max_observed_ds = series_df["ds"].max()
test_start = max_observed_ds - pd.DateOffset(months=TEST_HORIZON - 1)
train_backtest_df = series_df[series_df["ds"] < test_start].copy()
test_actuals = series_df[
    (series_df["ds"] >= test_start) & (series_df["ds"] <= max_observed_ds)
][["unique_id", "ds", "y"]].copy()

backtest_predictions = forecast_with_fallback(
    model_input_df=train_backtest_df,
    start_date=test_start,
    end_date=max_observed_ds,
    static_features=static_features,
)
evaluation_df = test_actuals.merge(
    backtest_predictions.rename(columns={"product_range": "unique_id"}),
    on=["unique_id", "ds"],
    how="left",
)
evaluation_df["ape"] = (
    (evaluation_df["y"] - evaluation_df["organic_sales_forecast"]).abs()
    / evaluation_df["y"].replace(0, pd.NA)
)
overall_mape = evaluation_df["ape"].mean(skipna=True) * 100
mape_by_product = (
    evaluation_df.groupby("unique_id", as_index=False)["ape"].mean().assign(
        mape_pct=lambda df: df["ape"] * 100
    )[["unique_id", "mape_pct"]]
).sort_values("mape_pct")

print(f"Backtest window: {test_start.date()} to {max_observed_ds.date()}")
print(f"Overall MAPE (%): {overall_mape:.2f}")
mape_by_product
# %%
col='Sanitaire'
df = series_df[series_df['unique_id'] == col][['ds', 'y']]
fsct_df = backtest_predictions[backtest_predictions['product_range'] == col][['ds', 'organic_sales_forecast']]
df = pd.concat([df, fsct_df], ignore_index=True)
px.line(df, x='ds', y=['y', 'organic_sales_forecast'])
# %%
px.scatter(evaluation_df, x='y', y='organic_sales_forecast', color='unique_id')
# %%
# Forecast Apr-Dec 2026
fit_df = series_df[series_df["ds"] < FORECAST_START].copy()
future_predictions = forecast_with_fallback(
    model_input_df=fit_df,
    start_date=FORECAST_START,
    end_date=FORECAST_END,
    static_features=static_features,
)
future_predictions

# %%
px.line(
    future_predictions,
    x="ds",
    y="organic_sales_forecast",
    color="product_range",
    title="Organic Sales Forecast (Apr-Dec 2026) by Product Range",
)

# %%
col='PDC'
df = fit_df[fit_df['unique_id'] == col][['ds', 'y']]
fsct_df = future_predictions[future_predictions['product_range'] == col][['ds', 'organic_sales_forecast']]
df = pd.concat([df, fsct_df], ignore_index=True)
px.line(df, x='ds', y=['y', 'organic_sales_forecast'])
# %%
future_predictions.to_parquet('../data/fcst/future_predictions.parquet')
# %%
kpi_df = pd.read_csv(KPI_PATH)
df = kpi_df[kpi_df['PRODUCT_RANGE'] == 'PDC'].copy()
df
# %%
df['date'] = pd.to_datetime(df['MONTH_YEAR'], format='%m-%Y')
px.line(df, x='date', y=['TOTAL_SALES'])

# %%
