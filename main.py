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
from dostics_simulations import dostics_simulations
from relojq import relojq
from experimentos import experimento_logbook
import numpy as np



N = 100000
escalado = 1e0
mu = 1*escalado
std = 0.0125*escalado
t_fin = 10000*escalado
t_res = 0.1#std/2
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


def plotDobleTics(numTics):

	misBins = np.arange(0, t_fin/mu)
	
	fig, ax10 = subplots()
	ax10.hist(numTics, bins=misBins)  
	ax10.set_ylabel(r"Histogram")
	ax10.set_xlabel(r"Number of tics")
	for item in ([ax10.title, ax10.xaxis.label, ax10.yaxis.label] + ax10.get_xticklabels() + ax10.get_yticklabels()):
	      item.set_fontsize(12)
	ax10.xaxis.set_major_locator(MaxNLocator(6))
	ax10.yaxis.set_major_locator(MaxNLocator(6))
	ax10.xaxis.set_minor_locator(AutoMinorLocator())
	ax10.yaxis.set_minor_locator(AutoMinorLocator())
	ax10.tick_params(axis='both', which='both', direction="in")
	ax10.set_title(r"Average number of tics: "+str(np.mean(numTics)))
	#ax10.set_ylim(0, 1)
	#ax10.set_xlim(0.5, len(lista_clicks)+0.5)
	savefig("results/esperaDobleTic_"+dist+"_test_"+str(escalado)+"_masTest.pdf",bbox_inches='tight')
	#show()





def plotter(x,y,titulo):
	fig, ax10 = subplots()
	ax10.plot(x,y)
	ax10.plot(x,y,'x')
	ax10.hlines(y=(mu/std)**2,xmin=x[0],xmax=x[-1], linestyle = '--', color='r')
	ax10.set_ylabel(r"Computed $R^{(1)}$")
	ax10.set_xlabel(r"Number of Clocks")
	ax10.grid()
	for item in ([ax10.title, ax10.xaxis.label, ax10.yaxis.label] + ax10.get_xticklabels() + ax10.get_yticklabels()):
	      item.set_fontsize(12)
	ax10.xaxis.set_major_locator(MaxNLocator(6))
	ax10.yaxis.set_major_locator(MaxNLocator(6))
	ax10.xaxis.set_minor_locator(AutoMinorLocator())
	ax10.yaxis.set_minor_locator(AutoMinorLocator())
	ax10.set_title(r"$\mu$: "+str(mu)+"  $\sigma$: "+str(std)+"  $t_0$: "+str(t_fin))
	#ax10.set_title("Average: "+str(mu)+"  Std: "+str(std)+"  N:"+str(N))
	ax10.tick_params(axis='both', which='both', direction="in")
	ax10.set_xlim(x[0],x[-1])
	ax10.set_ylim(4/0.25,4*0.25)
	ax10.set_xscale('log')
	#ax10.set_yscale('log')
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


	'''
	ss = sweep_simulations()
	
	Ns = [2,4,8,16,32,64,128,256,512,1024]#,2048]
	#Rs = ss.sweep_N(Ns, t_fin, mu, std)
	R = np.load('outN.npy')
	plotter(Ns, R[0,:], "N_sweep")
	'''

	'''
	t0s = [4,8,16,32,64,128,256,512,1024]
	#Rs = ss.sweep_t0(N, t0s, mu, std)
	R = np.load('outt.npy')
	plotter(t0s, R[1,:], "t0_sweep")
	'''





	
	#prueba doble tick no converge
	nsim = N
	t0 = t_fin

	dt = dostics_simulations()
	ks = dt.dostics(nsim, t0, mu, std, dist)
	mm = np.mean(ks[:,2])+np.mean(ks[:,4]**2)/std**2+np.mean(2*std*np.sqrt(ks[:,0])*ks[:,1]*ks[:,4])/std**2-np.mean(ks[:,4])**2/std**2
	print(" ")
	print(" ")
	print("**********************")

	print("Debug")
	print(np.mean(ks[:,4]))

	print(" ")
	print(np.mean((ks[:,5]-mu*ks[:,4])**2))
	#print(np.mean(ks[:,5]**2) + mu**2/2 - 2*mu*np.sum((ks[:,4]==1)*ks[:,5])/len(ks[:,5]))
	print(np.nanmean((np.sqrt(ks[:,0])*std*ks[:,1])**2)/2)

	print(" ")
	print("Real:", np.mean(ks[:,0]))
	print("Teoría tiempo parada:", np.mean((ks[:,5]-mu*ks[:,4])**2)/std**2)
	print("Desarrollo teoría antes de TLC:", np.mean(ks[:,5]**2) + mu**2/2 - 2*mu*np.sum((ks[:,4]==1)*ks[:,5])/len(ks[:,5]))
	print("Mi intento:", np.nanmean(ks[:,0]*ks[:,1]**2)/2)
	plotDobleTics(ks[:,0])
	

if __name__ == "__main__":
	main()




