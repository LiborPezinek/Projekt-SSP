import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acovf
from scipy.signal import periodogram
from scipy.special import inv_boxcox
import scipy.stats as stats
import pmdarima as pm



def LoadCsvData(csvFilePath = None) -> tuple[pd.Series, pd.Series]:
	df = pd.read_csv(csvFilePath)

    # Extract time components and values
	time_cols = df.iloc[:, 2:5]
	values = pd.to_numeric(df.iloc[:, 5], errors="coerce")

    # Combine time components into a single datetime column
	time_text = time_cols.astype(str).agg(" ".join, axis=1)
	time = pd.to_datetime(time_text, errors="coerce", dayfirst=False)

    # Validate that time and values are properly parsed
	valid = time.notna() & values.notna()
	if not valid.all():
		raise ValueError("Invalid rows detected with missing or unparseable data")

	return time, values

def VisualiseData(time, values, ylabel, title) -> None:
	plt.plot(time, values)
	plt.xlabel("Datum")
	plt.ylabel(ylabel)
	plt.title(title)
	plt.grid(True)
	plt.tight_layout()
	plt.show()

def VisualiseDataAndTrend(time, values, trend, ylabel, title) -> None:
		fig, ax = plt.subplots(figsize=(12, 6))
		ax.plot(time, values, label="Transformovaná data", linewidth=0.8)
		ax.plot(time[364:len(time)-365], trend[:len(trend)-365], label="Trend (365-denní klouzavý průměr)", linewidth=2.0)
		ax.set_xlabel("Date")
		ax.set_ylabel(ylabel)
		ax.set_title(title)
		ax.legend()
		ax.grid(True, alpha=0.3)
		fig.tight_layout()
		plt.show()

def CalcAndPlotACVF(time, values, title) -> None:
	acvf = acovf(values[366:], fft=True)
	lags = time[366:366+len(acvf)]

	# Plot positive and negative ACVF values with different colors.
	acvf_pos = np.where(acvf >= 0, acvf, np.nan)
	acvf_neg = np.where(acvf < 0, acvf, np.nan)
	plt.stem(lags, acvf_pos, linefmt="tab:blue", markerfmt="bo", basefmt="k-", label="ACVF >= 0")
	plt.stem(lags, acvf_neg, linefmt="tab:orange", markerfmt="o", basefmt=" ", label="ACVF < 0")

	plt.xlabel("Lag (dny)")
	plt.ylabel("Autocovariance")
	plt.title(title)
	plt.show()

def CalcAndPlotACVFwithPeriod(time, values, period, title) -> None:
	acvf = acovf(values[366:], fft=True)
	lags = time[366:366+len(acvf)]

	# Plot positive and negative ACVF values with different colors.
	acvf_pos = np.where(acvf >= 0, acvf, np.nan)
	acvf_neg = np.where(acvf < 0, acvf, np.nan)
	plt.stem(lags, acvf_pos, linefmt="tab:blue", markerfmt="bo", basefmt="k-", label="ACVF >= 0")
	plt.stem(lags, acvf_neg, linefmt="tab:orange", markerfmt="o", basefmt=" ", label="ACVF < 0")

	plt.xlabel("Lag (dny)")
	plt.ylabel("Autocovariance")
	plt.title(title)
	for idx, day in enumerate(pd.date_range(start=lags.iloc[3], end=lags.iloc[-1], freq=f"{period}D")):		# start at 3.1. to compensate for leap years
		if idx == 0: continue  # Skip the first vertical line at lag 0
		plt.axvline(
			x=day,
			color="red",
			linestyle="--",
			alpha=0.4,
			label=f"Každých {period} dní" if idx == 0 else None,
		)
	plt.legend()
	plt.show()

def CalcAndPlotRollingVariance(time, values, title) -> None:
	rollingVar = values.rolling(window=365).var()
	plt.plot(time, rollingVar)
	plt.title(title)
	plt.xlabel("Čas (dny)")
	plt.ylabel("Rozptyl")
	plt.show()

def CalcAndPlotPeriodogram(values, title) -> None:
	freqs, power = periodogram(values)
	periods = 1 / freqs

	plt.plot(periods, power)
	plt.xlabel("Perioda (dny)")
	plt.ylabel("Intenzita")
	plt.title(title)
	plt.show()

	return periods, power

def PlotQQ(values, title) -> None:
	fig = plt.figure(figsize=(6, 6))
	ax = fig.add_subplot(111)
	stats.probplot(values, dist="norm", plot=ax)
	ax.set_title(title)
	plt.show()

def FindOptimalArma(valuesDetrended):
	## Příliž velký dataset a perioda - hledá optimální model velmi dlouho, proto omezení na ARMA bez sezónní složky.
    # model = pm.auto_arima(
    #     valuesDetrended,
    #     d=0,               # force d=0 since series is stationary
    #     max_p=5, max_q=5,  # ARMA up to order 5
	# 	max_P=5, max_Q=5,  # seasonal ARMA up to order 2
    #     information_criterion="aic",
    #     stepwise=True,    # less exhaustive search
    #     seasonal=True, 
    #     m=365,             # period 365 days
    #     trace=True        # prints models being tested
    # )

	model = pm.auto_arima(
    	valuesDetrended,
    	start_p=0, max_p=5,
    	start_q=0, max_q=5,
    	seasonal=False,   # ARMA, information criterion will be calculated on non-seasonal model
    	information_criterion="aic",
    	stepwise=True,    # rychlejší hledání
    	trace=True        # vypíše kritéria modelů
	)
	return model

def PlotForecastARMA(valuesDetrendedSliced, forecast, conf_int, title) -> None:
	predictionIndices = np.arange(len(valuesDetrendedSliced) + 1, len(valuesDetrendedSliced) + len(forecast) + 1)
	plt.figure(figsize=(12, 5))
	plt.axhline(0, color="black", linewidth=0.8)
	plt.plot(valuesDetrendedSliced, label="Observed Data")
	plt.plot([predictionIndices[0] - 1, predictionIndices[0]], 
             [valuesDetrendedSliced.values[-1], forecast.values[0]], color="red")
	plt.plot(predictionIndices, forecast, color="red", label="Prediction")

	plt.fill_between(predictionIndices,
                     conf_int[:, 0],   # lower bound
                     conf_int[:, 1],   # upper bound
                     color="red", alpha=0.2, label="95% Confidence Interval")

	plt.title(title)
	plt.xlabel("Time [days]")
	plt.ylabel("")
	plt.legend()
	plt.grid(True, alpha=0.5)
	plt.tight_layout()
	plt.show()

def OrigDataPlotForecastARMA(values, preDiffLastVal, forecast, lam, title) -> None:
	valuesSliced = values[182:len(values)-182]

	predictionIndices = np.arange(len(valuesSliced) + 1, len(valuesSliced) + len(forecast) + 1)

	dediffForecast = np.concatenate([[preDiffLastVal], preDiffLastVal + np.cumsum(forecast)])
	forecastOrig = inv_boxcox(dediffForecast, lam)

	plt.figure(figsize=(12, 5))
	plt.axhline(0, color="black", linewidth=0.8)
	plt.plot(valuesSliced[:len(valuesSliced)-182], label="Observed Data")
	plt.plot([predictionIndices[0] - 1, predictionIndices[0]], 
             [valuesSliced[len(valuesSliced)-1], forecastOrig[0]], color="red")
	plt.plot(predictionIndices, forecastOrig[:len(predictionIndices)], color="red", label="Prediction")

	plt.title(title)
	plt.xlabel("Time [days]", fontsize=12)
	plt.ylabel("Flow", fontsize=12)
	plt.legend()
	plt.grid(True, alpha=0.5)
	plt.tight_layout()
	plt.show()

def PlotForecastManual(mHat, sHat, values, valuesTransformedSliced, predictionTime, lam, title) -> None:
	valuesSliced = values[182:len(values)-182]

	predictionIndices = np.arange(len(valuesTransformedSliced) + 1, len(valuesTransformedSliced) + predictionTime + 1)

	prediction = np.zeros(len(predictionIndices))
	for i in range(len(prediction)):
		prediction[i] = mHat[(len(mHat)-1) -len(predictionIndices) + i] + sHat[(len(sHat)-1) -182 - len(predictionIndices) + i]
	predictionOrig = inv_boxcox(prediction, lam)

	plt.figure(figsize=(12, 5))
	plt.axhline(0, color="black", linewidth=0.8)
	plt.plot(valuesSliced[182:len(valuesSliced)-181], label="Original data")
	plt.plot([predictionIndices[0] - 1, predictionIndices[0]],
	         [valuesSliced[len(valuesSliced)-1], predictionOrig[0]], color="red")
	plt.plot(predictionIndices, predictionOrig, label="Prediction", color="red")
	plt.title(title)
	plt.xlabel("Time [days]", fontsize=12)
	plt.ylabel("Flow", fontsize=12)
	plt.legend()
	plt.grid(True, alpha=0.5)
	plt.tight_layout()
	plt.show()
