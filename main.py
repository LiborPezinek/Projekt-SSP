##  Závěrečný projekt z předmětu SSP
##  Autor: Libor Pezinek, 248222
## 	Data: Průtoková data ze stanice Vyšší Brod, řeka Vltava, v letech 2010-2025, získaná z databáze ČHMÚ (Český hydrometeorologický ústav)
##  Odkaz na data: https://isvs.chmi.cz/ords/f?p=11002:2:11142408395097:::RP,2:P2_SEQ:%5C116%5C
## Pozn.: Dataset zkrácen za účelem snížení výpočetní náročnosti, eliminace outlieru (povodně 2002) a eliminace strukturálních změn v povodí řeky.
## 		  Původní dataset obsahuje data od roku 1981 do roku 2025

import numpy as np
import pandas as pd
from scipy.stats import boxcox

import utils


def main() -> None:
	# Load data
	time, values = utils.LoadCsvData("data/QD_109000_Data.csv")

	## Předpočítané operace na datech, abychom mohli volat pouze příslušný blok kódu pro daný úkol
	valuesTransformed, lam = boxcox(values)						  		# Transform the data using Box-Cox transformation to stabilize variance
	valuesTransformed = pd.Series(valuesTransformed, index=values.index) # Convert back to pandas Series for easier handling
	valuesDetrended = valuesTransformed.diff().dropna() 			  		# Difference the data to detrend
	trend = valuesTransformed.rolling(window=365).mean().dropna()  		# Calculate the trend using a rolling mean with a window of 365 days (1 year)

	# 2:  Data vykreslete, posuďte autokovarianční funkci.
	# 3:  Ověřte, zda je třeba provést transformaci stabilizující rozptyl a případně ji proveďte.  
	# 4:  Odstraňte trend vhodnou metodou.
	# 5:  Identifikujte periodu sezónní složky (z podstaty dat, ověřte periodogramem)
	# 6:  Odhadněte trendovou a sezónní složku
	# 7:  Testem náhodnosti ověřte, zda získaná rezidua jsou IID a zda jsou normální
	# 8:  Vykreslete autokorelační funkci a parciální autokorelační funkci reziduí a pokuste se určit vhodný ARMA model
	# 9:  Určete predikci o h kroků dopředu buď užitím předchozích kroků nebo pomocí SARIMA.
	task = 6

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
		# valuesBoxCox, lam = boxcox(values)	# g(x) =  1 - 1/x
		print("Optimal lambda for Box-Cox transformation:", lam)

		# Plot log transformed data
		utils.VisualiseData(time, valuesTransformed, ylabel = "Průtok", title = "Transformovaná data ze stanice Vyšší Brod, řeka Vltava (2010-2025)")

		# Re-calculate and plot rolling variance for transformed data
		utils.CalcAndPlotRollingVariance(time, valuesTransformed, title = "Klouzavý rozptyl logaritmovaných průtokových dat")

	# 4:  Odstraňte trend vhodnou metodou.
	if task == 4:
		#  trend = valuesBoxCox.rolling(window=365).mean().dropna()
		# Show trend
		utils.VisualiseData(time[364:], trend, ylabel = "Trend", title = "Trend transformovaných průtokových dat ze stanice Vyšší Brod, řeka Vltava (2010-2025)")
		
		# Plot trend and data together
		utils.VisualiseDataAndTrend(time, valuesTransformed, trend, ylabel = "Průtok", title = "Transformovaná data a trend ze stanice Vyšší Brod, řeka Vltava (2010-2025)")

		# valuesDetrended = valuesBoxCox.diff().dropna()
		# Plot detrended data
		utils.VisualiseData(time[1:], valuesDetrended, ylabel = "Diferencovaná průtoková data", title = "Diferencovaná data ze stanice Vyšší Brod, řeka Vltava (2010-2025)")
	
		# Calculate and plot autocovariance function for seasonally differenced data
		utils.CalcAndPlotACVF(time[1:], valuesDetrended, title = "Autocovarianční funkce pro diferencovaná data")
		
		## Diferenciace lepší výsledky oproti odečtení moving average

	# 5:  Identifikujte periodu sezónní složky (z podstaty dat, ověřte periodogramem)
	if task == 5:
		# Calculate and plot periodogram to identify seasonality
		periods, power = utils.CalcAndPlotPeriodogram(valuesTransformed[365:], title = "Periodogram průtokových dat ze stanice Vyšší Brod, řeka Vltava (2010-2025)")
		peakPeriod = periods[np.argmax(power)]
		print(f"Peak period: {peakPeriod}")	# 365.25 - confirming the yearly seasonality

		## Diferencovaná data neukazují smysluplnou periodu, proto periodogram na pouze stabilizovaných datech (popř. na původních)

	# 6:  Odhadněte trendovou a sezónní složku
	if task == 6:
		# Centered moving average: 0.5*x[i] + x[i+1] + ... + x[i+period-1] + 0.5*x[i+period]
		mHat = valuesTransformed[182:len(valuesTransformed)-182].rolling(window=365, center=True).mean().dropna()
		sHat = utils.CalcSeasonalComponent(valuesTransformed[364:len(valuesTransformed)-364], mHat, period=365)

		print("Size of mHat:", len(mHat), "Size of sHat:", len(sHat))

		utils.VisualiseData(time[364:len(mHat) + 364], mHat, ylabel = "Trend", title = "Trend průtokových dat ze stanice Vyšší Brod, řeka Vltava (2010-2025)")
		utils.VisualiseData(time[364:len(sHat) + 364], sHat, ylabel = "Sezónní složka", title = "Sezónní složka průtokových dat ze stanice Vyšší Brod, řeka Vltava (2010-2025)")


if __name__ == "__main__":
	main()
