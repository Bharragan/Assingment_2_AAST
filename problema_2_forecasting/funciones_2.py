import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
#adfuller
from statsmodels.tsa.stattools import adfuller
# Para Prophet
from prophet import Prophet
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ===============================
#     UTILS PARA FIGURAS
# ===============================
def _get_fig_dir():
    """Devuelve la carpeta /figuras relativa a este archivo."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(base_dir, "figuras")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


# ===============================
#   (a) GRAFICAR SERIE MÉTRICA
# ===============================
def plot_time_series(ts, title="Serie de Tiempo"):
    fig_dir = _get_fig_dir()

    plt.figure(figsize=(10, 4))
    plt.plot(ts, linewidth=1)
    plt.title(title)
    plt.xlabel("Tiempo")
    plt.ylabel("Valor")
    plt.grid(True)

    fname = f"time_series_{title.replace(' ', '_')}.png"
    fpath = os.path.join(fig_dir, fname)
    plt.savefig(fpath, dpi=300)
    plt.show()
    plt.close()

    return fpath  # útil para imprimir rutas en main
    


# ===============================
#   (b) PRUEBA DE ESTACIONARIDAD ADF
# ===============================
def adf_check(ts):
    """
    Ejecuta ADF y retorna un diccionario con:
    - estadístico
    - p-value
    - lags usados
    - tamaño de la muestra
    - valores críticos
    """
    result = adfuller(ts)
    
    output = {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "lags used": result[2],
        "n_obs": result[3],
        "critical values": result[4]
    }

    return output
    

# ===============================
#   ANÁLISIS COMPLETO EN UNA FUNCIÓN
# ===============================
def exploratory_analysis(ts, nombre="serie"):
    """
    Ejecuta:
    1) Gráfico
    2) Prueba ADF
    """
    ruta_figura = plot_time_series(ts, title=nombre)
    adf_res = adf_check(ts)

    return {
        "figura_guardada": ruta_figura,
        "adf": adf_res
    }


# ===============================
#     UTILS PARA FIGURAS
# ===============================
def _get_fig_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(base_dir, "figuras")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


# ===============================
#     MÉTRICAS DE PREDICCIÓN
# ===============================
def forecast_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


# ===============================
#     GRAFICAR PREDICCIÓN
# ===============================
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


# ===============================
#        MODELO ARIMA
# ===============================
def train_arima(ts, order=(1,1,1)):
    model = ARIMA(ts, order=order)
    fitted = model.fit()
    pred = fitted.fittedvalues
    metrics = forecast_metrics(ts, pred)
    fig_path = plot_forecast(ts, pred, title=f"ARIMA_{order}")
    return {"model": fitted, "forecast": pred, "metrics": metrics, "figure": fig_path}


# ===============================
#       MODELO SARIMA
# ===============================
def train_sarima(ts, order=(1,1,1), seasonal_order=(0,1,1,12)):
    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    pred = fitted.fittedvalues
    metrics = forecast_metrics(ts, pred)
    fig_path = plot_forecast(ts, pred, title=f"SARIMA_{order}_{seasonal_order}")
    return {"model": fitted, "forecast": pred, "metrics": metrics, "figure": fig_path}


# ===============================
#      HOLT-WINTERS
# ===============================
def train_holtwinters(ts, seasonal_periods=12, trend='add', seasonal='add'):
    model = ExponentialSmoothing(ts, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    fitted = model.fit()
    pred = fitted.fittedvalues
    metrics = forecast_metrics(ts, pred)
    fig_path = plot_forecast(ts, pred, title=f"HoltWinters")
    return {"model": fitted, "forecast": pred, "metrics": metrics, "figure": fig_path}


from sklearn.ensemble import RandomForestRegressor

def train_random_forest(ts, n_lags=12, n_estimators=100, random_state=42):
    X, y = [], []
    for i in range(n_lags, len(ts)):
        X.append(ts.values[i-n_lags:i])
        y.append(ts.values[i])
    
    X = np.array(X)
    y = np.array(y)
    
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    
    # Predicción recursiva (rolling)
    y_pred_full = list(ts.values[:n_lags])
    for i in range(n_lags, len(ts)):
        X_input = np.array(y_pred_full[i-n_lags:i]).reshape(1, -1)
        y_pred_full.append(model.predict(X_input)[0])
    
    y_pred_full = np.array(y_pred_full)
    
    # ⚡ Alineamos con el índice de ts
    y_pred_series = pd.Series(y_pred_full, index=ts.index)
    
    metrics = forecast_metrics(ts.values, y_pred_series.values)
    fig_path = plot_forecast(ts, y_pred_series, title=f"RandomForest_{n_lags}lags")
    
    return {"model": model, "forecast": y_pred_series, "metrics": metrics, "figure": fig_path}

from statsmodels.graphics.tsaplots import plot_acf

# ===============================
# Entrenar varias configuraciones de ARIMA
# ===============================
def train_multiple_arima(ts, orders):
    """
    orders: lista de tuplas (p,d,q)
    Retorna lista de resultados con métricas, gráficas y residuales
    """
    results = []
    fig_dir = _get_fig_dir()

    for order in orders:
        res = train_arima(ts, order=order)
        fitted = res['model']
        residuals = fitted.resid

        # --- Graficar residuales ---
        plt.figure(figsize=(10,4))
        plt.plot(residuals)
        plt.title(f"Residuales ARIMA{order}")
        plt.grid(True)
        fname_res = f"residuals_ARIMA_{order}.png"
        fpath_res = os.path.join(fig_dir, fname_res)
        plt.savefig(fpath_res, dpi=300)
        plt.close()

        # --- Graficar ACF de residuales ---
        fig, ax = plt.subplots(figsize=(10,4))
        plot_acf(residuals, lags=30, ax=ax)
        plt.title(f"ACF de residuales ARIMA{order}")
        fname_acf = f"ACF_residuals_ARIMA_{order}.png"
        fpath_acf = os.path.join(fig_dir, fname_acf)
        plt.savefig(fpath_acf, dpi=300)
        plt.close()

        # Guardar resultados
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
            "figure_residuals_acf": fpath_acf
        })

    return results

# ===============================
# Resumen comparativo de métricas ARIMA
# ===============================
def summarize_arima(results):
    """
    Genera un DataFrame resumen de AIC, BIC y log-likelihood
    """
    summary = []
    for r in results:
        summary.append({
            "order": r['order'],
            "AIC": r['AIC'],
            "BIC": r['BIC'],
            "loglikelihood": r['loglikelihood'],
            "RMSE": r['forecast_metrics']['RMSE'],
            "MAE": r['forecast_metrics']['MAE'],
            "MAPE": r['forecast_metrics']['MAPE']
        })
    return pd.DataFrame(summary)


def evaluate_models_last_year(ts, freq='MS', n_lags_rf=12):
    """
    Evalúa ARIMA, SARIMA, Holt-Winters y RandomForest usando el último año como test.
    ts: pandas Series con índice datetime
    freq: frecuencia temporal, por defecto 'MS' (Month Start)
    n_lags_rf: número de lags para RandomForest
    """
    # ----------------------------
    # 1. Separar train/test
    # ----------------------------
    last_year = ts.index.year.max()
    train = ts[ts.index.year < last_year]
    test = ts[ts.index.year == last_year]

    results = []

    # ----------------------------
    # 2. ARIMA (ejemplo con (1,1,1))
    # ----------------------------
    arima_res = train_arima(train, order=(1,1,1))
    # Forecast sobre test
    arima_forecast = arima_res['model'].forecast(steps=len(test))
    arima_metrics = forecast_metrics(test.values, arima_forecast)
    results.append({
        "model": "ARIMA(1,1,1)",
        "MAE": arima_metrics['MAE'],
        "RMSE": arima_metrics['RMSE'],
        "MAPE": arima_metrics['MAPE'],
        "forecast": arima_forecast
    })

    # ----------------------------
    # 3. SARIMA (ejemplo con (1,1,1),(0,1,1,12))
    # ----------------------------
    sarima_res = train_sarima(train, order=(1,1,1), seasonal_order=(0,1,1,12))
    sarima_forecast = sarima_res['model'].forecast(steps=len(test))
    sarima_metrics = forecast_metrics(test.values, sarima_forecast)
    results.append({
        "model": "SARIMA(1,1,1)(0,1,1,12)",
        "MAE": sarima_metrics['MAE'],
        "RMSE": sarima_metrics['RMSE'],
        "MAPE": sarima_metrics['MAPE'],
        "forecast": sarima_forecast
    })

    # ----------------------------
    # 4. Holt-Winters
    # ----------------------------
    hw_res = train_holtwinters(train, seasonal_periods=12)
    # Forecast out-of-sample
    hw_model = hw_res['model']
    hw_forecast = hw_model.forecast(steps=len(test))
    hw_metrics = forecast_metrics(test.values, hw_forecast)
    results.append({
        "model": "Holt-Winters",
        "MAE": hw_metrics['MAE'],
        "RMSE": hw_metrics['RMSE'],
        "MAPE": hw_metrics['MAPE'],
        "forecast": hw_forecast
    })

    # ----------------------------
    # 5. RandomForest
    # ----------------------------
    rf_res = train_random_forest(train, n_lags=n_lags_rf)
    # Forecast recursivo sobre test
    y_pred = list(train.values[-n_lags_rf:])  # iniciar con últimos lags del train
    for val in test.values:
        X_input = np.array(y_pred[-n_lags_rf:]).reshape(1,-1)
        y_pred.append(rf_res['model'].predict(X_input)[0])
    rf_forecast = np.array(y_pred[n_lags_rf:])  # solo predicciones sobre test
    rf_metrics = forecast_metrics(test.values, rf_forecast)
    results.append({
        "model": f"RandomForest_{n_lags_rf}lags",
        "MAE": rf_metrics['MAE'],
        "RMSE": rf_metrics['RMSE'],
        "MAPE": rf_metrics['MAPE'],
        "forecast": rf_forecast
    })

    # ----------------------------
    # 6. DataFrame resumen
    # ----------------------------
    metrics_df = pd.DataFrame([{
        "Model": r["model"],
        "MAE": r["MAE"],
        "RMSE": r["RMSE"],
        "MAPE": r["MAPE"]
    } for r in results])

    return metrics_df, results, train, test


def plot_comparative_forecasts_last_year(ts, results, train, test, title="Comparación de Pronósticos Último Año"):
    """
    ts: serie completa
    results: lista de diccionarios con claves 'model' y 'forecast' (solo forecast sobre test)
    train, test: series de entrenamiento y validación
    """
    fig_dir = _get_fig_dir()
    plt.figure(figsize=(12,6))

    # Graficar serie real del último año
    plt.plot(test.index, test.values, label="Real", color='black', linewidth=2)

    # Graficar cada forecast
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for i, r in enumerate(results):
        plt.plot(test.index, r['forecast'], label=r['model'], color=colors[i % len(colors)], linestyle='--', linewidth=2)

    plt.title(title)
    plt.xlabel("Tiempo")
    plt.ylabel("Valor")
    plt.grid(True)
    plt.legend()
    
    fname = f"comparative_forecasts_last_year.png"
    fpath = os.path.join(fig_dir, fname)
    plt.savefig(fpath, dpi=300)
    plt.show()
    plt.close()

    return fpath


def plot_forecast_with_confidence(ts, results, train, test, alpha=0.05, title="Pronóstico con Intervalos de Confianza"):
    """
    ts: serie completa
    results: lista de diccionarios con claves 'model' y 'forecast'
    train, test: series de entrenamiento y validación
    alpha: nivel de significancia (default 0.05 para IC 95%)
    """
    fig_dir = _get_fig_dir()
    plt.figure(figsize=(12,6))

    # Serie real
    plt.plot(test.index, test.values, label="Real", color='black', linewidth=2)

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, r in enumerate(results):
        model_name = r['model']
        forecast = r['forecast']

        # Intentamos calcular intervalos de confianza si es posible
        try:
            fitted_model = r.get('model', None)
            if "ARIMA" in model_name or "SARIMA" in model_name:
                pred_res = fitted_model.get_forecast(steps=len(test))
                mean_pred = pred_res.predicted_mean
                conf_int = pred_res.conf_int(alpha=alpha)
                lower = conf_int.iloc[:,0]
                upper = conf_int.iloc[:,1]
            elif "Holt-Winters" in model_name:
                # Aproximamos IC usando desviación estándar de residuales
                se = np.std(fitted_model.resid)
                mean_pred = forecast
                lower = mean_pred - 1.96*se
                upper = mean_pred + 1.96*se
            else:  # RandomForest
                # Aproximación simple usando desviación de árboles
                estimators_preds = np.array([tree.predict(fitted_model.X_train_[-len(test):]) for tree in fitted_model.estimators_])
                lower = np.percentile(estimators_preds, 2.5, axis=0)
                upper = np.percentile(estimators_preds, 97.5, axis=0)
                mean_pred = forecast
        except:
            # Si no se puede calcular IC, dibujamos solo forecast
            mean_pred = forecast
            lower = forecast
            upper = forecast

        # Graficar línea y sombreado
        plt.plot(test.index, mean_pred, label=model_name, color=colors[i % len(colors)], linestyle='--', linewidth=2)
        plt.fill_between(test.index, lower, upper, color=colors[i % len(colors)], alpha=0.2)

    plt.title(title)
    plt.xlabel("Tiempo")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)

    fname = "forecast_with_confidence.png"
    fpath = os.path.join(fig_dir, fname)
    plt.savefig(fpath, dpi=300)
    plt.show()
    plt.close()

    return fpath


def sensitivity_arima(ts, orders_list):
    """
    ts: serie temporal
    orders_list: lista de tuplas (p,d,q) a probar
    Retorna DataFrame con métricas para cada configuración
    """
    results = []

    for order in orders_list:
        res = train_arima(ts, order=order)
        fitted = res['model']
        metrics = res['metrics']
        results.append({
            "order": order,
            "RMSE": metrics['RMSE'],
            "MAE": metrics['MAE'],
            "MAPE": metrics['MAPE'],
            "AIC": fitted.aic,
            "BIC": fitted.bic
        })
        # Graficar forecast
        plt.figure(figsize=(10,4))
        plt.plot(ts, label="Original")
        plt.plot(res['forecast'], linestyle='--', label=f"ARIMA{order}")
        plt.title(f"Sensibilidad ARIMA{order}")
        plt.grid(True)
        plt.legend()
        plt.show()

    return pd.DataFrame(results)


def sensitivity_holtwinters(ts, seasonal_periods_list, trend='add', seasonal='add'):
    """
    ts: serie temporal
    seasonal_periods_list: lista de valores de frecuencia estacional a probar
    Retorna DataFrame con métricas y genera gráficas
    """
    results = []

    for sp in seasonal_periods_list:
        res = train_holtwinters(ts, seasonal_periods=sp, trend=trend, seasonal=seasonal)
        metrics = res['metrics']
        results.append({
            "seasonal_periods": sp,
            "RMSE": metrics['RMSE'],
            "MAE": metrics['MAE'],
            "MAPE": metrics['MAPE']
        })
        # Graficar forecast
        plt.figure(figsize=(10,4))
        plt.plot(ts, label="Original")
        plt.plot(res['forecast'], linestyle='--', label=f"Holt-Winters s={sp}")
        plt.title(f"Sensibilidad Holt-Winters s={sp}")
        plt.grid(True)
        plt.legend()
        plt.show()

    return pd.DataFrame(results)
