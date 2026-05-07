from time import time

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import acovf


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

def Visualise(time, values, ylabel, title) -> None:
	plt.figure(figsize=(10, 5))
	plt.plot(time, values)
	plt.xlabel("Datum")
	plt.ylabel(ylabel)
	plt.title(title)
	plt.grid(True)
	plt.tight_layout()
	plt.show()

def PlotAutocovariance(time, values) -> None:
	acvf = acovf(values, fft=True)
	lags = time[:len(acvf[:5600])]
	plt.stem(lags, acvf[:5600])
	plt.xlabel("Lag (days)")
	plt.ylabel("Autocovariance")
	plt.title("Autocovarianční funkce pro průtoková data")
	plt.show()

def PlotRollingVariance(time, values, title) -> None:
	rollingVar = values.rolling(window=365).var()
	plt.plot(time, rollingVar)
	plt.title(title)
	plt.xlabel("Time (days)")
	plt.ylabel("Variance")
	plt.show()
