##  Závěrečný projekt z předmětu SSP
##  Autor: Libor Pezinek, 248222
## 	Data: Průtoková data ze stanice Vyšší Brod, řeka Vltava, v letech 2010-2025, získaná z databáze ČHMÚ (Český hydrometeorologický ústav)
##  Odkaz na data: https://isvs.chmi.cz/ords/f?p=11002:2:11142408395097:::RP,2:P2_SEQ:%5C116%5C
## Pozn.: Dataset zkrácen za účelem snížení výpočetní náročnosti, eliminace outlieru (povodně 2002) a eliminace strukturálních změn v povodí řeky.
## 		  Původní dataset obsahuje data od roku 1981 do roku 2025

import numpy as np

import utils


def main() -> None:
	time, values = utils.LoadCsvData("data/QD_109000_Data.csv")

	# Log-transform the data to stabilize variance
	valuesLog = np.log(values)

	# Difference the data to detrend (and also remove seasonality)
	valuesDetrended = values.diff().dropna()

	# Calculate the trend using a rolling mean with a window of 365 days (1 year)
	trend = valuesLog.rolling(window=365).mean().dropna()	# just for visualization purposes, not used in further analysis


	# # Visualize original data
	# utils.VisualiseData(time, values, ylabel = "Průtok (m³/s)", title = "Průtoková data ze stanice Vyšší Brod, řeka Vltava (2010-2025)")
	
	# # Calculate and plot autocovariance function
	# utils.CalcAndPlotACVF(time, values, title = "Autocovarianční funkce pro průtoková data")

	# # Calculate and plot rolling variance - to know whether to transform or not
	# utils.CalcAndPlotRollingVariance(time, values, title = "Klouzavý rozptyl průtokových dat")

	# # Plot log transformed data
	# utils.VisualiseData(time, valuesLog, ylabel = "Logaritmus průtoku (log(m³/s))", title = "Transformovaná data (logaritmus) ze stanice Vyšší Brod, řeka Vltava (2010-2025)")

	# # Re-calculate and plot rolling variance for transformed data
	# utils.CalcAndPlotRollingVariance(time, valuesLog, title = "Klouzavý rozptyl logaritmovaných průtokových dat")

	# # Show trend
	# utils.VisualiseData(time[365:], trend, ylabel = "Trend (log(m³/s))", title = "Trend logaritmovaných průtokových dat ze stanice Vyšší Brod, řeka Vltava (2010-2025)")

	# # Plot detrended data
	# utils.VisualiseData(time[1:], valuesDetrended, ylabel = "Diferencovaná průtoková data (log(m³/s))", title = "Sezónně diferencovaná data ze stanice Vyšší Brod, řeka Vltava (2010-2025)")
	
	# # Calculate and plot autocovariance function for seasonally differenced data
	# utils.CalcAndPlotACVF(time[1:], valuesDetrended, title = "Autocovarianční funkce pro diferencovaná data")

	# # # Calculate and plot autocovariance function with period annotation
	periods, power = utils.CalcAndPlotPeriodogram(values, title = "Periodogram průtokových dat ze stanice Vyšší Brod, řeka Vltava (2010-2025)")
	peakPeriod = periods[np.argmax(power)]
	print(peakPeriod)	# 365.25 - confirming the yearly seasonality



if __name__ == "__main__":
	main()
