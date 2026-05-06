##  Závěrečný projekt z předmětu SSP
##  Autor: Libor Pezinek, 248222
## 	Data: Průtoková data ze stanice Vyšší Brod, řeka Vltava, v letech 1981-2025, získaná z databáze ČHMÚ (Český hydrometeorologický ústav)
##  Odkaz na data: https://isvs.chmi.cz/ords/f?p=11002:2:11142408395097:::RP,2:P2_SEQ:%5C116%5C

import matplotlib.pyplot as plt

from utils.visualisation import Visualise, LoadCsvData
from statsmodels.graphics.tsaplots import plot_acf


def main() -> None:
	# Load data and visualise
	time, values = LoadCsvData("data/QD_109000_Data.csv")
	Visualise(time, values)
	
	# Plot ACF
	fig  = plot_acf(values, lags=1000)
	plt.xlabel("Lag")
	plt.ylabel("Autokorelace")
	plt.show()



if __name__ == "__main__":
	main()
