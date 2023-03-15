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

	def calcula_estadisticos(self, N, registro):
		numTics = np.zeros(N, dtype=float)
		for i in range(N):
			numTics[i] = registro[i][-1]

		mediaK = np.mean(numTics)
		stdK = np.std(numTics)

		#mediaK_1clk = mediaK
		#stdK_1clk = stdK/np.sqrt(N)

		#ejeX_1clk = np.linspace(int(mediaK_1clk)-3*int(stdK_1clk), int(mediaK_1clk)+3*int(stdK_1clk), 100)
		#fdpK_1clk = stats.norm.pdf(ejeX, mediaK_1clk, stdK_1clk)
		
		ejeX = np.linspace(int(mediaK)-3*int(stdK), int(mediaK)+3*int(stdK), 100)
		fdpK = stats.norm.pdf(ejeX, mediaK, stdK)

		return(mediaK, stdK, ejeX, fdpK)



	def resuelveR(self, mediaK_1clk, stdK_1clk, ejeX, fdpK, t_fin, params):

		k1 = ejeX[int(len(ejeX)/2)]
		pk1 = fdpK[int(len(ejeX)/2)]
		k2 = ejeX[int(len(ejeX)/3)]
		pk2 = fdpK[int(len(ejeX)/3)]

		def func(x):
			return [pk1 - 0.5*(erf(  1/(np.sqrt(2)*np.sqrt(k1)) * (x[0]-k1*x[1]) )-erf(  1/(np.sqrt(2)*np.sqrt(k1+1)) * (x[0]-(k1+1)*x[1]) )),
					pk2 - 0.5*(erf(  1/(np.sqrt(2)*np.sqrt(k2)) * (x[0]-k2*x[1]) )-erf(  1/(np.sqrt(2)*np.sqrt(k2+1)) * (x[0]-(k2+1)*x[1]) )),]

		root = fsolve(func, [t_fin/params[1], params[0]/params[1]])
		return(root)



	def sweep_N(self, Ns, t0, mu, std):

		params = [mu, std]
		t_fin = 501
		t_res = std/2
		dist = "normal"

		Rs = np.zeros((2,len(Ns)), dtype = float)
		i = 0
		t = np.arange(0, t_fin, t_res, dtype = float)
		
		for N in Ns:			
			exp = experimento_logbook(t, N, dist, params)
			exp.ini_RelojesSimultaneos()
			[t_tics, intervalos, registro, lista_clicks] = exp.run()
			[mediaK, stdK, ejeX, fdpK] = self.calcula_estadisticos(N, registro)
			Rs[:,i] = self.resuelveR(mediaK, stdK, ejeX, fdpK, t_fin, params)
			i+=1

		np.save('outN.npy', Rs)
    

		return(Rs)


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

