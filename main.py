##  Závěrečný projekt z předmětu SSP
##  Autor: Libor Pezinek, 248222
## 	Data: Průtoková data ze stanice Vyšší Brod, řeka Vltava, v letech 2010-2025, získaná z databáze ČHMÚ (Český hydrometeorologický ústav)
##  Odkaz na data: https://isvs.chmi.cz/ords/f?p=11002:2:11142408395097:::RP,2:P2_SEQ:%5C116%5C
## Pozn.: Dataset zkrácen za účelem snížení výpočetní náročnosti, eliminace outlieru (povodně 2002) a eliminace strukturálních změn v povodí řeky.
## 		  Původní dataset obsahuje data od roku 1981 do roku 2025

import numpy as np

import utils


def main() -> None:
	# Load data and visualise
	time, values = utils.LoadCsvData("data/QD_109000_Data.csv")
	utils.Visualise(time, values, ylabel = "Průtok (m³/s)", title = "Průtoková data ze stanice Vyšší Brod, řeka Vltava (2010-2025)")
	
	# Calculate and plot autocovariance function
	utils.PlotAutocovariance(time, values)

	# Calculate and plot rolling variance - to know whether to transform or not
	utils.PlotRollingVariance(time, values, title = "Klouzavý rozptyl průtokových dat")

	# Log transform and plot
	valuesLog = np.log(values)
	utils.Visualise(time, valuesLog, ylabel = "Logaritmus průtoku (log(m³/s))", title = "Transformovaná data (logaritmus) ze stanice Vyšší Brod, řeka Vltava (2010-2025)")

	utils.PlotRollingVariance(time, valuesLog, title = "Klouzavý rozptyl logaritmovaných průtokových dat")

	


if __name__ == "__main__":
	main()
