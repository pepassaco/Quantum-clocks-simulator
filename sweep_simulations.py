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


class sweep_simulations:

	def printProgressBar(self,value,label):
		n_bar = 40 #size of progress bar
		max = 100
		j= value/max
		sys.stdout.write('\r')
		bar = '█' * int(n_bar * j)
		bar = bar + '-' * int(n_bar * (1-j))

		sys.stdout.write(f"{label.ljust(10)} | [{bar:{n_bar}s}] {int(100 * j)}% ")
		sys.stdout.flush()





	def calcula_estadisticos(self, N, registro):
		numTics = np.zeros(N, dtype=float)
		for i in range(N):
			numTics[i] = registro[i][-1]

		mediaK = np.mean(numTics)
		stdK = np.std(numTics)

		if(N == 1024):
			self.plotKs(numTics, mediaK, stdK)
		return(mediaK, stdK)


	def plotKs(self, numTics, mediaK, stdK):


		misBins = np.arange(int(mediaK)-3*int(stdK), int(mediaK)+3*int(stdK))
		ejeX = np.linspace(int(mediaK)-3*int(stdK), int(mediaK)+3*int(stdK), 100)

		fig, ax10 = subplots()
		ax10.hist(numTics, bins=misBins, density = True)  # arguments are passed to np.histogram
		ax10.plot(ejeX, stats.norm.pdf(ejeX, mediaK, stdK))
		ax10.set_ylabel(r"Density")
		ax10.set_xlabel(r"Number of tics")
		for item in ([ax10.title, ax10.xaxis.label, ax10.yaxis.label] + ax10.get_xticklabels() + ax10.get_yticklabels()):
		      item.set_fontsize(12)
		ax10.xaxis.set_major_locator(MaxNLocator(6))
		ax10.yaxis.set_major_locator(MaxNLocator(6))
		ax10.xaxis.set_minor_locator(AutoMinorLocator())
		ax10.yaxis.set_minor_locator(AutoMinorLocator())
		ax10.tick_params(axis='both', which='both', direction="in")
		ax10.set_title(r"std: "+str(stdK))
		#ax10.set_ylim(0, 1)
		#ax10.set_xlim(0.5, len(lista_clicks)+0.5)
		show()


	def sweep_N(self, Ns, t0, mu, std):

		params = [mu, std]
		t_fin = t0
		t_res = std/2
		dist = "normal"

		Rs = np.zeros((2,len(Ns)), dtype = float)
		i = 0
		t = np.arange(0, t_fin, t_res, dtype = float)
		
		for N in Ns:			
			exp = experimento_logbook(t, N, dist, params)
			exp.ini_RelojesSimultaneos()
			[t_tics, intervalos, registro, lista_clicks] = exp.run()
			[mediaK, stdK] = self.calcula_estadisticos(N, registro)
			Rs[:,i] = [mediaK/stdK**2, 0]

			print("-----")
			print(mediaK, stdK)
			print(N, Rs[0,i])
			i+=1
			
			#self.printProgressBar(int(i/nsim*100), "Completado")

		np.save('outN.npy', Rs)
    

		return(Rs)





















	#AUN POR CORREGIR
	def sweep_t0(self, N, t0s, mu, std):

		params = [mu, std]
		
		t_res = std/2
		dist = "normal"

		Rs = np.zeros((2,len(t0s)), dtype = float)

		i=0
		for t_fin in t0s:
		
			t = np.arange(0, t_fin, t_res, dtype = float)
			exp = experimento_logbook(t, N, dist, params)
			exp.ini_RelojesSimultaneos()
			[t_tics, intervalos, registro, lista_clicks] = exp.run()
			[mediaK, stdK, ejeX, fdpK] = self.calcula_estadisticos(N, registro)
			Rs[:,i] = self.resuelveR(mediaK, stdK, ejeX, fdpK, t_fin, params)
			i+=1
			print(t_fin)
		np.save('outt.npy', Rs)
		return(Rs)


	def sweep_2D(self, Ns, t0s, mu, std):

		params = [mu, std]
		
		t_res = std/2
		dist = "normal"

		Rs = np.zeros((2,len(Ns),len(t0s)), dtype = float)

		i = 0
		
		for N in Ns:
			j = 0
			for t_fin in t0s:
				t = np.arange(0, t_fin, t_res, dtype = float)
				exp = experimento_logbook(t, N, dist, params)
				exp.ini_RelojesSimultaneos()
				[t_tics, intervalos, registro, lista_clicks] = exp.run()
				[mediaK, stdK, ejeX, fdpK] = self.calcula_estadisticos(N, registro)
				Rs[:,i,j] = self.resuelveR(mediaK, stdK, ejeX, fdpK, t_fin, params)
				j+=1
			i+=1
		return(Rs)

