##  Závěrečný projekt z předmětu SSP
##  Autor: Libor Pezinek, 248222
## 	Data: Průtoková data ze stanice Vyšší Brod, řeka Vltava, v letech 2010-2025, získaná z databáze ČHMÚ (Český hydrometeorologický ústav)
##  Odkaz na data: https://isvs.chmi.cz/ords/f?p=11002:2:11142408395097:::RP,2:P2_SEQ:%5C116%5C
## Pozn.: Dataset zkrácen za účelem snížení výpočetní náročnosti, eliminace outlieru (povodně 2002) a eliminace strukturálních změn v povodí řeky.
## 		  Původní dataset obsahuje data od roku 1981 do roku 2025

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from scipy.stats import boxcox

import utils


def main() -> None:
	# Load data
	time, values = utils.LoadCsvData("data/QD_109000_Data.csv")

	# Log-transform the data to stabilize variance
	valuesLog = np.log(values)

	# Difference the data to detrend (and also remove seasonality)
	valuesDetrended = values.diff().dropna()

	# Calculate the trend using a rolling mean with a window of 365 days (1 year)
	trend = valuesLog.rolling(window=365).mean().dropna()

	# 2:  Data vykreslete, posuďte autokovarianční funkci.
	# 3:  Ověřte, zda je třeba provést transformaci stabilizující rozptyl a případně ji proveďte.  
	# 4:  Odstraňte trend vhodnou metodou.
	# 5:  Identifikujte periodu sezónní složky (z podstaty dat, ověřte periodogramem)
	# 6:  Odhadněte trendovou a sezónní složku
	# 7:  Testem náhodnosti ověřte, zda získaná rezidua jsou IID a zda jsou normální
	# 8:  Vykreslete autokorelační funkci a parciální autokorelační funkci reziduí a pokuste se určit vhodný ARMA model
	# 9:  Určete predikci o h kroků dopředu buď užitím předchozích kroků nebo pomocí SARIMA.
	task = 3

	# 2:  Data vykreslete, posuďte autokovarianční funkci.
	if task == 2:
		# Visualize original data
		utils.VisualiseData(time, values, ylabel = "Průtok (m³/s)", title = "Průtoková data ze stanice Vyšší Brod, řeka Vltava (2010-2025)")
	
		# Calculate and plot autocovariance function
		utils.CalcAndPlotACVF(time, values, title = "Autocovarianční funkce pro průtoková data")

	# 3:  Ověřte, zda je třeba provést transformaci stabilizující rozptyl a případně ji proveďte.  
	if task == 3:
		# Calculate and plot rolling variance - to know whether to transform or not
		utils.CalcAndPlotRollingVariance(time, values, title = "Klouzavý rozptyl průtokových dat")

		# Box-Cox transformation to find optimal lambda for variance stabilization
		valuesBoxCox, lam = boxcox(values)	# g(x) =  1 - 1/x
		print("Optimal lambda for Box-Cox transformation:", lam)
		valuesBoxCox = pd.Series(valuesBoxCox, index=values.index)		# Convert back to pandas Series for easier handling

		# Plot log transformed data
		utils.VisualiseData(time, valuesBoxCox, ylabel = "Průtok", title = "Transformovaná data ze stanice Vyšší Brod, řeka Vltava (2010-2025)")

		# Re-calculate and plot rolling variance for transformed data
		utils.CalcAndPlotRollingVariance(time, valuesBoxCox, title = "Klouzavý rozptyl logaritmovaných průtokových dat")

	# 4:  Odstraňte trend vhodnou metodou.
	if task == 4:
		# Show trend
		utils.VisualiseData(time[365:], trend, ylabel = "Trend (log(m³/s))", title = "Trend logaritmovaných průtokových dat ze stanice Vyšší Brod, řeka Vltava (2010-2025)")

		# Plot detrended data
		utils.VisualiseData(time[1:], valuesDetrended, ylabel = "Diferencovaná průtoková data (log(m³/s))", title = "Diferencovaná data ze stanice Vyšší Brod, řeka Vltava (2010-2025)")
	
		# Calculate and plot autocovariance function for seasonally differenced data
		utils.CalcAndPlotACVF(time[1:], valuesDetrended, title = "Autocovarianční funkce pro diferencovaná data")

	# 5:  Identifikujte periodu sezónní složky (z podstaty dat, ověřte periodogramem)
	if task == 5:
		# Calculate and plot periodogram to identify seasonality
		periods, power = utils.CalcAndPlotPeriodogram(values, title = "Periodogram průtokových dat ze stanice Vyšší Brod, řeka Vltava (2010-2025)")
		peakPeriod = periods[np.argmax(power)]
		print(peakPeriod)	# 365.25 - confirming the yearly seasonality

	# 6:  Odhadněte trendovou a sezónní složku
	if task == 6:
		decompTSR = seasonal_decompose(valuesLog, model='additive', period=365)
		decompTSR.plot()
		plt.show()



if __name__ == "__main__":
	main()
