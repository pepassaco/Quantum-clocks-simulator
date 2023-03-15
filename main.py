from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from matplotlib.cm import coolwarm, ScalarMappable
from matplotlib import gridspec
from matplotlib.pyplot import axhline, subplots, show, hist, figure, setp, colorbar, plot, cm, title, xlabel, ylabel, grid, legend, savefig, axes, pcolormesh, close
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, MaxNLocator
import matplotlib.colors as colors
import scipy.stats as stats
from scipy.optimize import fsolve
from scipy.special import erf
from sweep_simulations import sweep_simulations

from relojq import relojq
from experimentos import experimento_logbook
import numpy as np



N = 500
mu = 1.0
std = 0.5
t_fin = 101
t_res = std/2
dist = "normal"
tipo = "coso" 			# sim frac coso
params = [mu, std]





def plotter_registro(t, N, registro):


	rels_axis = np.linspace(1,N,N)
	

	fig, ax10 = subplots()

	fig.set_figheight(6)
	fig.set_figwidth(6)

	miplot1 = ax10.pcolormesh(t, rels_axis, registro, vmin=0, cmap='magma', alpha=1, antialiased=True)
	miplot1.set_edgecolor('face')

	ax10.set_ylabel(r"Clock number")
	ax10.set_xlabel(r"Time")
	for item in ([ax10.title, ax10.xaxis.label, ax10.yaxis.label] + ax10.get_xticklabels() + ax10.get_yticklabels()):
	      item.set_fontsize(12)
	ax10.xaxis.set_major_locator(MaxNLocator(6))
	ax10.yaxis.set_major_locator(MaxNLocator(6))
	ax10.xaxis.set_minor_locator(AutoMinorLocator())
	ax10.yaxis.set_minor_locator(AutoMinorLocator())
	ax10.tick_params(axis='both', which='both', direction="in")

	#ax10.set_xlim(263,266)
	#ax10.set_ylim(1.6, 3.8)
	fig.colorbar(miplot1)

	savefig("results/reg_"+tipo+"_tf="+str(t_fin)+"_tr="+str(t_res)+"_N="+str(N)+"_fdp_"+dist+"_μ="+str(params[0])+"_σ="+str(params[1])+".pdf",bbox_inches='tight')
	show()
	close()



def plotter_lista_clicks(N, lista_clicks):

	ejeX = np.linspace(1,len(lista_clicks),len(lista_clicks))

	fig, ax10 = subplots()
	fig.set_figheight(3)
	fig.set_figwidth(14)
	miplot1 = ax10.pcolormesh(ejeX, [0, 1], [lista_clicks, lista_clicks], vmin=0, vmax=N-1, cmap='gist_ncar', alpha=1, antialiased=True)
	miplot1.set_edgecolor('face')
	ax10.set_aspect(0.05*len(lista_clicks))
	ax10.set_ylabel(r" ")
	ax10.set_xlabel(r"Ticking registry")
	ax10.set_yticklabels([])
	ax10.set_xticklabels([])
	ax10.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left = False)
	ax10.set_ylim(0, 1)
	ax10.set_xlim(0.5, len(lista_clicks)+0.5)
	#fig.colorbar(miplot1)

	savefig("results/list_"+tipo+"_tf="+str(t_fin)+"_tr="+str(t_res)+"_N="+str(N)+"_fdp_"+dist+"_μ="+str(params[0])+"_σ="+str(params[1])+".pdf",bbox_inches='tight')
	show()
	close()


def plotKs(numTics, mediaK, stdK):


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
	#ax10.set_ylim(0, 1)
	#ax10.set_xlim(0.5, len(lista_clicks)+0.5)
	savefig("results/kDist_t0="+str(t_fin)+"_N="+str(N)+"_fdp_"+dist+"_μ="+str(params[0])+"_σ="+str(params[1])+".pdf",bbox_inches='tight')
	#show()






def resuelveR(mediaK_1clk, stdK_1clk, ejeX, fdpK):

	k1 = ejeX[int(len(ejeX)/2)]
	pk1 = fdpK[int(len(ejeX)/2)]
	k2 = ejeX[int(len(ejeX)/3)]
	pk2 = fdpK[int(len(ejeX)/3)]

	def func(x):
		return [pk1 - 0.5*(erf(  1/(np.sqrt(2)*np.sqrt(k1)) * (x[0]-k1*x[1]) )-erf(  1/(np.sqrt(2)*np.sqrt(k1+1)) * (x[0]-(k1+1)*x[1]) )),
				pk2 - 0.5*(erf(  1/(np.sqrt(2)*np.sqrt(k2)) * (x[0]-k2*x[1]) )-erf(  1/(np.sqrt(2)*np.sqrt(k2+1)) * (x[0]-(k2+1)*x[1]) )),]

	root = fsolve(func, [t_fin/params[1], params[0]/params[1]])
	return(root)



def plotter(x,y,titulo):
	fig, ax10 = subplots()
	ax10.plot(x,y)
	ax10.hlines(y=mu/std,xmin=x[0],xmax=x[-1], linestyle = '--', color='r')
	ax10.set_ylabel(r"Computed $R^{(1)}$")
	ax10.set_xlabel(r"Number of Clocks")
	ax10.grid()
	for item in ([ax10.title, ax10.xaxis.label, ax10.yaxis.label] + ax10.get_xticklabels() + ax10.get_yticklabels()):
	      item.set_fontsize(12)
	ax10.xaxis.set_major_locator(MaxNLocator(6))
	ax10.yaxis.set_major_locator(MaxNLocator(6))
	ax10.xaxis.set_minor_locator(AutoMinorLocator())
	ax10.yaxis.set_minor_locator(AutoMinorLocator())
	ax10.set_title("Average: "+str(mu)+"  Std: "+str(std)+"  t_0:"+str(t_fin))
	ax10.tick_params(axis='both', which='both', direction="in")
	ax10.set_xlim(x[0],x[-1])
	ax10.set_ylim(2*2/3,2*3/2)
	#ax10.set_xscale('log')
	ax10.set_yscale('log')
	
	savefig("results/"+titulo+".pdf",bbox_inches='tight')
	show()

def main():

	''' 
	#LOGBOOK
	t = np.arange(0, t_fin, t_res, dtype = float)
	exp = experimento_logbook(t, N, dist, params)

	match tipo:
		case "sim":
			exp.ini_RelojesSimultaneos()
		case "frac":
			exp.ini_RelojesFraccion()
		case "coso":
			exp.ini_RelojesConUnoSuelto()


	[t_tics, intervalos, registro, lista_clicks] = exp.run()
	plotter_registro(t, N, registro)
	plotter_lista_clicks(N, lista_clicks)
	'''
	'''
	#LIMCENT

	print(" ")
	print("*******************************************")
	print("**********QUANTUM CLOCK SIMULATOR**********")
	print("*******************************************")
	print(" ")
	print("N:", N, "t_0:", t_fin)
	print(" ")
	print("*******************************************")
	t = np.arange(0, t_fin, t_res, dtype = float)
	exp = experimento_logbook(t, N, dist, params)

	exp.ini_RelojesSimultaneos()
	

	[t_tics, intervalos, registro, lista_clicks] = exp.run()

	#ESTO ESTA ROTO
	[mediaK_1clk, stdK_1clk, ejeX, fdpK] = calcula_estadisticos(N, registro)

	print(" ")
	print("Average number of tics:", mediaK_1clk)
	print("Std in the number of tics:", stdK_1clk)
	print(" ")
	print("*******************************************")


	sols = resuelveR(mediaK_1clk, stdK_1clk, ejeX, fdpK)

	print(" ")
	print("Valores reales:",[t_fin/params[1], params[0]/params[1]])
	print("Valores obtenidos:", sols)
	print(" ")
	print("*******************************************")
	print(" ")
	'''


	
	ss = sweep_simulations()
	Ns = [1,2,4,8,16,32,64,128,256,512]#, 1024, 2048]

	Rs = ss.sweep_N(Ns, t_fin, mu, std)

	R = np.load('outN.npy')
	plotter(Ns, R[1,:], "N_sweep")

if __name__ == "__main__":
	main()




