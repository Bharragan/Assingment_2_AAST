import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- MODELOS ---
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- TESTS ---
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

# --- ML EXTRA ---
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
# from prophet import Prophet   # si lo usas, actívalo


# ==========================================================
#  1. UTILS GENERALES (figuras, métricas)
# ==========================================================

def _get_fig_dir():
    """Devuelve la carpeta /figuras relativa a este archivo."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(base_dir, "figuras")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def forecast_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


# ==========================================================
#  2. EXPLORACIÓN (plot + ADF)
# ==========================================================

def plot_time_series(ts, title="Serie de Tiempo"):
    fig_dir = _get_fig_dir()
    plt.figure(figsize=(10, 4))
    plt.plot(ts, linewidth=1)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

    fname = f"time_series_{title.replace(' ', '_')}.png"
    fpath = os.path.join(fig_dir, fname)
    plt.savefig(fpath, dpi=300)
    plt.show()
    plt.close()
    return fpath


def adf_check(ts):
    result = adfuller(ts)
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "lags used": result[2],
        "n_obs": result[3],
        "critical values": result[4]
    }



# ==========================================================
#  3. MODELOS INDIVIDUALES (ARIMA, SARIMA, HW, RF)
# ==========================================================

def plot_forecast(y_true, y_pred, title="Pronóstico"):
    fig_dir = _get_fig_dir()
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label="Original", linewidth=2)
    plt.plot(y_pred, label="Predicción", linestyle="--")
    plt.title(title)
    plt.xlabel("Tiempo")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)

    fname = f"forecast_{title.replace(' ', '_')}.png"
    fpath = os.path.join(fig_dir, fname)
    plt.savefig(fpath, dpi=300)
    plt.close()
    return fpath


# --------------------
# 3.1 ARIMA
# --------------------
def train_arima(ts, order=(1,1,1)):
    model = ARIMA(ts, order=order)
    fitted = model.fit()
    pred = fitted.fittedvalues
    metrics = forecast_metrics(ts, pred)
    fig_path = plot_forecast(ts, pred, title=f"ARIMA_{order}")
    return {
        "model": fitted,
        "forecast": pred,
        "metrics": metrics,
        "figure": fig_path
    }


# --------------------
# 3.2 SARIMA
# --------------------
def train_sarima(ts, order=(1,1,1), seasonal_order=(0,1,1,12)):
    model = SARIMAX(
        ts,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted = model.fit(disp=False)
    pred = fitted.fittedvalues
    metrics = forecast_metrics(ts, pred)
    fig_path = plot_forecast(ts, pred, title=f"SARIMA_{order}_{seasonal_order}")
    return {
        "model": fitted,
        "forecast": pred,
        "metrics": metrics,
        "figure": fig_path
    }


# --------------------
# 3.3 Holt-Winters
# --------------------
def train_holtwinters(ts, seasonal_periods=12, trend='add', seasonal='add'):
    model = ExponentialSmoothing(
        ts,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    )
    fitted = model.fit()
    pred = fitted.fittedvalues
    metrics = forecast_metrics(ts, pred)
    fig_path = plot_forecast(ts, pred, title=f"HoltWinters")
    return {
        "model": fitted,
        "forecast": pred,
        "metrics": metrics,
        "figure": fig_path
    }


# --------------------
# 3.4 RandomForest
# --------------------
def train_random_forest(ts, n_lags=12, n_estimators=100, random_state=42):

    X, y = [], []
    for i in range(n_lags, len(ts)):
        X.append(ts.values[i-n_lags:i])
        y.append(ts.values[i])
    
    X = np.array(X)
    y = np.array(y)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X, y)
    
    # Predicción rolling
    y_pred_full = list(ts.values[:n_lags])
    for i in range(n_lags, len(ts)):
        X_input = np.array(y_pred_full[i-n_lags:i]).reshape(1, -1)
        y_pred_full.append(model.predict(X_input)[0])
    
    y_pred_full = np.array(y_pred_full)
    y_pred_series = pd.Series(y_pred_full, index=ts.index)
    
    metrics = forecast_metrics(ts.values, y_pred_series.values)
    fig_path = plot_forecast(ts, y_pred_series, title=f"RandomForest_{n_lags}lags")
    
    return {
        "model": model,
        "forecast": y_pred_series,
        "metrics": metrics,
        "figure": fig_path
    }


# ==========================================================
#  4. VARIACIÓN DE PARÁMETROS ARIMA (residuales + AIC/BIC)
# ==========================================================

def train_multiple_arima(ts, orders):
    results = []
    fig_dir = _get_fig_dir()

    for order in orders:
        res = train_arima(ts, order)
        fitted = res['model']
        residuals = fitted.resid

        # Plot residuales
        plt.figure(figsize=(10,4))
        plt.plot(residuals)
        plt.title(f"Residuales ARIMA{order}")
        plt.grid(True)
        fname_res = f"residuals_ARIMA_{order}.png"
        fpath_res = os.path.join(fig_dir, fname_res)
        plt.savefig(fpath_res, dpi=300)
        plt.close()

        # ACF residuales
        fig, ax = plt.subplots(figsize=(10,4))
        plot_acf(residuals, lags=30, ax=ax)
        fname_acf = f"ACF_residuals_ARIMA_{order}.png"
        plt.savefig(os.path.join(fig_dir, fname_acf), dpi=300)
        plt.close()

        results.append({
            "order": order,
            "model": fitted,
            "forecast_metrics": res['metrics'],
            "figure_forecast": res['figure'],
            "AIC": fitted.aic,
            "BIC": fitted.bic,
            "loglikelihood": fitted.llf,
            "residuals": residuals,
            "figure_residuals": fpath_res,
            "figure_residuals_acf": os.path.join(fig_dir, fname_acf)
        })

    return results


def summarize_arima(results):
    rows = []
    for r in results:
        rows.append({
            "order": r["order"],
            "AIC": r["AIC"],
            "BIC": r["BIC"],
            "loglikelihood": r["loglikelihood"],
            "RMSE": r["forecast_metrics"]["RMSE"],
            "MAE": r["forecast_metrics"]["MAE"],
            "MAPE": r["forecast_metrics"]["MAPE"]
        })
    return pd.DataFrame(rows)


# ==========================================================
#  5. VALIDACIÓN (último año)
# ==========================================================

def evaluate_models_last_year(ts, n_lags_rf=12):

    last_year = ts.index.year.max()
    train = ts[ts.index.year < last_year]
    test = ts[ts.index.year == last_year]

    results = []

    # ARIMA
    arima_res = train_arima(train, order=(1,1,1))
    arima_fore = arima_res["model"].forecast(steps=len(test))
    arima_m = forecast_metrics(test.values, arima_fore.values)
    results.append({"model": "ARIMA(1,1,1)", "forecast": arima_fore, **arima_m})

    # SARIMA
    sarima_res = train_sarima(train)
    sarima_fore = sarima_res["model"].forecast(steps=len(test))
    sarima_m = forecast_metrics(test.values, sarima_fore.values)
    results.append({"model": "SARIMA(1,1,1)(0,1,1,12)", "forecast": sarima_fore, **sarima_m})

    # Holt-Winters
    hw_res = train_holtwinters(train)
    hw_fore = hw_res["model"].forecast(steps=len(test))
    hw_m = forecast_metrics(test.values, hw_fore.values)
    results.append({"model": "Holt-Winters", "forecast": hw_fore, **hw_m})

    # RandomForest
    rf_res = train_random_forest(train, n_lags=n_lags_rf)
    preds = list(train.values[-n_lags_rf:])
    for val in test.values:
        X_input = np.array(preds[-n_lags_rf:]).reshape(1, -1)
        preds.append(rf_res["model"].predict(X_input)[0])
    rf_fore = np.array(preds[n_lags_rf:])
    rf_m = forecast_metrics(test.values, rf_fore)
    results.append({"model": "RandomForest", "forecast": rf_fore, **rf_m})

    metrics_df = pd.DataFrame(results)[["model","MAE","RMSE","MAPE"]]
    return metrics_df, results, train, test


# ==========================================================
#  6. COMPARACIÓN GRÁFICA E INTERVALOS
# ==========================================================

def plot_comparative_forecasts_last_year(ts, results, train, test):
    fig_dir = _get_fig_dir()
    plt.figure(figsize=(12,6))

    plt.plot(test.index, test.values, label="Real", color="black", linewidth=2)
    colors = ["blue","red","green","orange","purple"]

    for i, r in enumerate(results):
        plt.plot(test.index, r["forecast"], label=r["model"],
                 color=colors[i % len(colors)], linestyle="--", linewidth=2)

    plt.legend()
    plt.grid(True)
    plt.title("Comparación de Pronósticos Último Año")
    fpath = os.path.join(fig_dir, "comparative_forecasts_last_year.png")
    plt.savefig(fpath, dpi=300)
    plt.show()
    plt.close()
    return fpath


def plot_forecast_with_confidence(ts, results, train, test, alpha=0.05):
    fig_dir = _get_fig_dir()
    plt.figure(figsize=(12,6))
    plt.plot(test.index, test.values, label="Real", color="black", linewidth=2)

    colors = ["blue","red","green","orange","purple"]

    for i, r in enumerate(results):
        model_name = r["model"]
        forecast = r["forecast"]

        # ARIMA / SARIMA IC
        if hasattr(r, "model") and ("ARIMA" in model_name or "SARIMA" in model_name):
            pred_res = r["model"].get_forecast(steps=len(test))
            mean_pred = pred_res.predicted_mean
            conf = pred_res.conf_int(alpha=alpha)
            lower, upper = conf.iloc[:,0], conf.iloc[:,1]
        else:
            # fallback: sin IC
            mean_pred, lower, upper = forecast, forecast, forecast

        plt.plot(test.index, mean_pred, color=colors[i], linestyle="--", label=model_name)
        plt.fill_between(test.index, lower, upper, color=colors[i], alpha=0.2)

    plt.title("Pronóstico con Intervalos de Confianza")
    plt.legend()
    plt.grid(True)

    fpath = os.path.join(fig_dir, "forecast_with_confidence.png")
    plt.savefig(fpath, dpi=300)
    plt.show()
    plt.close()
    return fpath


# ==========================================================
#  7. SENSIBILIDAD
# ==========================================================

def sensitivity_arima(ts, orders_list):
    results = []
    for order in orders_list:
        res = train_arima(ts, order)
        fitted = res["model"]
        metrics = res["metrics"]
        results.append({
            "order": order,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"],
            "AIC": fitted.aic,
            "BIC": fitted.bic
        })
    return pd.DataFrame(results)


def sensitivity_holtwinters(ts, seasonal_periods_list, trend='add', seasonal='add'):
    results = []
    for sp in seasonal_periods_list:
        res = train_holtwinters(ts, seasonal_periods=sp)
        m = res["metrics"]
        results.append({
            "seasonal_periods": sp,
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
            "MAPE": m["MAPE"]
        })
    return pd.DataFrame(results)
