from time import time

import matplotlib.pyplot as plt
import pandas as pd


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

def Visualise(time, values) -> None:
	plt.figure(figsize=(10, 5))
	plt.plot(time, values)
	plt.xlabel("Datum")
	plt.ylabel("Průtok (m³/s)")
	plt.title("Průtoková data ze stanice Vyšší Brod, řeka Vltava (1981-2025)")
	plt.grid(True)
	plt.tight_layout()
	plt.show()
