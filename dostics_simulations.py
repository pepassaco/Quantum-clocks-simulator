from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from matplotlib.cm import coolwarm, ScalarMappable
from matplotlib import gridspec
from matplotlib.pyplot import subplots, show, hist, figure, setp, colorbar, plot, cm, title, xlabel, ylabel, grid, legend, savefig, axes, pcolormesh, close
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, MaxNLocator
import matplotlib.colors as colors
import scipy.stats as stats
from scipy.optimize import fsolve
from scipy.special import erf

from relojq import relojq
from experimentos import experimento_logbook
import numpy as np
import sys


class dostics_simulations:

	def printProgressBar(self,value,label):
		n_bar = 40 #size of progress bar
		max = 100
		j= value/max
		sys.stdout.write('\r')
		bar = '█' * int(n_bar * j)
		bar = bar + '-' * int(n_bar * (1-j))

		sys.stdout.write(f"{label.ljust(10)} | [{bar:{n_bar}s}] {int(100 * j)}% ")
		sys.stdout.flush()

	def sacaNc(self, t, mu, std):
		out = 0
		#print(" ")
		#print("Umbral:", mu/std/2/np.sqrt(t))
		if(t % 2 == 0):
			while (out > -mu/std/2/np.sqrt(t)):
				out = (np.random.normal(0, 1))
			#print("Salida:", out)
			return(out)
		else:
			while (out < mu/std/2/np.sqrt(t)):
				out = (np.random.normal(0, 1))
			#print("Salida:", out)
			return(np.nan)#std*np.sqrt(t)*out)



	def dostics(self, nsim, t0, mu, std, dist):

		params = [mu, std]
		t_fin = t0
		t_res = std/2

		t = np.arange(0, t_fin, t_res, dtype = float)
		results = np.zeros((nsim,6), dtype = float)

		for i in range(nsim):
			exp = experimento_logbook(t, 2, dist, params)
			exp.ini_RelojesFraccion()
			[t_tics, intervalos, registro, lista_clicks] = exp.run()


			dobleTic = False
			j = 2 #ignoro el primer tic determinista de cada reloj

			#print(lista_clicks)

			while not dobleTic:
				if(lista_clicks[j] == lista_clicks[j+1]):

					if(lista_clicks[j+1] == 0):
						results[i,5] = t_tics[0][int(j/2+1)]-(t_tics[1][int(j/2)]-mu/2)
						#print(results[i,5]<mu/2)
					else:
						#print(len(t_tics[0]))
						#print(int(j/2+1))
						results[i,5] = t_tics[0][int(j/2+1)]-(t_tics[1][int(j/2+1)]-mu/2)
						#print(results[i,5]>mu/2)


					# tic en j=2 => t1+t3 < mu/2+t2

					results[i,0] = j+1
					
					rng = self.sacaNc(j+1,mu,std)
					results[i,1] = rng					
					results[i,4] = j%2 == 0 # Vale 1 si dt el reloj A
					dobleTic = True

				elif(j >= len(lista_clicks)-2):
					dobleTic = True
				else:
					j+=1


			self.printProgressBar(int(i/nsim*100), "Completado")
			#print("Reloj:", j%2, "Suma:", results[i,5], "Media:", results[i,4], "Suma tilde:", results[i,5]-results[i,4])

    

		return(results)