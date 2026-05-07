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

	# Visualize original data
	# utils.VisualiseData(time, values, ylabel = "Průtok (m³/s)", title = "Průtoková data ze stanice Vyšší Brod, řeka Vltava (2010-2025)")
	
	# Calculate and plot autocovariance function
	# utils.CalcAndPlotAutocovariance(time, values)

	# Calculate and plot rolling variance - to know whether to transform or not
	# utils.CalcAndPlotRollingVariance(time, values, title = "Klouzavý rozptyl průtokových dat")

	# Log transform and plot
	valuesLog = np.log(values)
	# utils.VisualiseData(time, valuesLog, ylabel = "Logaritmus průtoku (log(m³/s))", title = "Transformovaná data (logaritmus) ze stanice Vyšší Brod, řeka Vltava (2010-2025)")

	# Re-calculate and plot rolling variance for transformed data
	# utils.CalcAndPlotRollingVariance(time, valuesLog, title = "Klouzavý rozptyl logaritmovaných průtokových dat")

	# show trend
	trend = valuesLog.rolling(window=730).mean()
	utils.VisualiseData(time, trend, ylabel = "Trend (log(m³/s))", title = "Trend logaritmovaných průtokových dat ze stanice Vyšší Brod, řeka Vltava (2010-2025)")

	


if __name__ == "__main__":
	main()
