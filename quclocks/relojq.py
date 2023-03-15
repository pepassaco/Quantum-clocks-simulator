import numpy as np

class relojq():

	def __init__(self, fdp, t, t0=0):

		self.t = t
		self.t0 = t0
		self.fdp = fdp


	def run(self):

		fin_t = False
		t_tics = [0]
		intervalos = [0]


		while(not fin_t):									#Deja correr el reloj mientras no se pase del tiempo que le metimos

			aleatorio = self.fdp()

			t_tics.append(t_tics[-1]+aleatorio)
			intervalos.append(aleatorio)


			if(t_tics[-1] >= self.t[-1]):					#Cuando se acaba, elimina la muestra sobrante que no entró el tiempo y el 0 del principio
				fin_t = True
				t_tics.remove(t_tics[-1])
				intervalos.remove(intervalos[-1])
				t_tics.remove(t_tics[0])
				intervalos.remove(intervalos[0])

		

		for i in range(len(t_tics)):						#Metemos aqui t0 (primer interv más largo y todos los tics se retrasan)
			t_tics[i]+=self.t0

		intervalos[0]+=self.t0



		registro = np.zeros(len(self.t), dtype = float)		#Simulamos nuestro registro
		ntic = 0


		for i in range(len(self.t)):
			if(i > 0 and ntic < len(t_tics) and self.t[i] >= t_tics[ntic]):
				ntic+=1
			registro[i] = ntic




		return(t_tics, intervalos, registro)