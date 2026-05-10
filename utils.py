from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acovf
from scipy.signal import periodogram
from scipy.stats import levene


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

def CalcAndPlotACVF(time, values, title) -> None:
	acvf = acovf(values, fft=True)
	lags = time[:len(acvf)]

	# Plot positive and negative ACVF values with different colors.
	acvf_pos = np.where(acvf >= 0, acvf, np.nan)
	acvf_neg = np.where(acvf < 0, acvf, np.nan)
	plt.stem(lags, acvf_pos, linefmt="tab:blue", markerfmt="bo", basefmt="k-", label="ACVF >= 0")
	plt.stem(lags, acvf_neg, linefmt="tab:orange", markerfmt="o", basefmt=" ", label="ACVF < 0")

	plt.xlabel("Lag (days)")
	plt.ylabel("Autocovariance")
	plt.title(title)
	plt.show()

def CalcAndPlotACVFwithPeriod(time, values, period, title) -> None:
	acvf = acovf(values, fft=True)
	lags = time[:len(acvf)]

	# Plot positive and negative ACVF values with different colors.
	acvf_pos = np.where(acvf >= 0, acvf, np.nan)
	acvf_neg = np.where(acvf < 0, acvf, np.nan)
	plt.stem(lags, acvf_pos, linefmt="tab:blue", markerfmt="bo", basefmt="k-", label="ACVF >= 0")
	plt.stem(lags, acvf_neg, linefmt="tab:orange", markerfmt="o", basefmt=" ", label="ACVF < 0")

	plt.xlabel("Lag (days)")
	plt.ylabel("Autocovariance")
	plt.title(title)
	for idx, day in enumerate(pd.date_range(start=lags.iloc[3], end=lags.iloc[-1], freq=f"{period}D")):		# start at 3.1. to compensate for leap years
		if idx == 0: continue  # Skip the first vertical line at lag 0
		plt.axvline(
			x=day,
			color="red",
			linestyle="--",
			alpha=0.4,
			label=f"Every {period} days" if idx == 0 else None,
		)
	plt.legend()
	plt.show()

def CalcAndPlotRollingVariance(time, values, title) -> None:
	rollingVar = values.rolling(window=365).var()
	plt.plot(time, rollingVar)
	plt.title(title)
	plt.xlabel("Time (days)")
	plt.ylabel("Variance")
	plt.show()

def LeveneTest(values1, values2) -> None:
		# H0: The variances of the two groups are equal (no need for transformation)
		# H1: The variances of the two groups are not equal (transformation may be needed)
		_, p = levene(values1, values2)		# Test for equal variances between 2 sets of data
		print(p)	# If p-value is less than 0.05, we reject the null hypothesis of equal variances
		print  ("Variances are equal, no transformation needed" if p >= 0.05 else "Variances are not equal, transformation may be needed")
	

def CalcAndPlotPeriodogram(values, title) -> None:
	freqs, power = periodogram(values)
	periods = 1 / freqs

	plt.plot(periods, power)
	plt.xlabel("Period (days)")
	plt.ylabel("Power")
	plt.title(title)
	plt.show()

	return periods, power
