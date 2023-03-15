from relojq import relojq
import numpy as np

class experimento_logbook():

	def __init__(self, t, N=10, dist="normal", params=[1.0, 0.01]):

		self.t = t
		self.N = N
		self.params = params

		match dist:
			case "normal":
				self.fdp = lambda: np.random.default_rng().normal(params[0], params[1])
			case"beta":
				self.fdp = lambda: np.random.default_rng().beta(params[0], params[1])
			case "exponential":
				self.fdp = lambda: np.random.default_rng().exponential(params[0])
			case "poisson":
				self.fdp = lambda: np.random.default_rng().poisson(params[0])
			case "triangular":
				self.fdp = lambda: np.random.default_rng().exponential(params[0], params[1], params[2])
			case _:
				print("Error con la fdp. Saliendo...")
				return()

		self.relojesQ = []



	def ini_RelojesSimultaneos(self):

		t0s = np.zeros(self.N, dtype=float)
		for i in range(self.N):
			self.relojesQ.append(relojq(self.fdp, self.t, t0s[i]))

	def ini_RelojesFraccion(self):

		t0s = np.linspace(0,self.params[0],self.N)		#inicializo un reloj cada media(tic)/N tiempo
		for i in range(self.N):
			self.relojesQ.append(relojq(self.fdp, self.t, t0s[i]))

	def ini_RelojesConUnoSuelto(self):
		ref = relojq(self.fdp, self.t, 0)
		[t_tics, intervalos, registro] = ref.run()
		if(len(t_tics)>self.N):
			for i in range(self.N):
				self.relojesQ.append(relojq(self.fdp, self.t, t_tics[i]))





	def run(self):

		
		t_tics = []
		intervalos = []
		registro = []

		for reloj in self.relojesQ:
			[t_ticsR, intervalosR, registroR] = reloj.run()

			
			t_tics.append(t_ticsR)
			intervalos.append(intervalosR)
			registro.append(registroR)


		
		t_clics_totales = sorted([item for sublist in t_tics for item in sublist])
		lista_clicks = []

		for i in range(len(t_clics_totales)):
			lista_clicks.append(int(self.in_list(t_clics_totales[i], t_tics)))


		return(t_tics, intervalos, registro, lista_clicks)


	def in_list(self, c, classes):
	    for i, sublist in enumerate(classes):
	        if c in sublist:
	            return i
	    return -1
