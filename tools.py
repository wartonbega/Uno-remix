from math import *
import numpy as np

class vector2():
	def __init__(self, dx, dy):
		self.dx = dx
		self.dy = dy
		self.v = np.array([dx, dy])

	def Scalaire(self, vec2):
		return (self.dx * vec2.dx) + (self.dy * vec2.dy)

	def Hortogonal(self):
		return vector2(-self.dy, self.dx)

	def Norme(self):
		return sqrt(self.dx ** 2 + self.dy **2)
	
	def Unitaire(self, new=False):
		"""Si new = True:
			renvoie un nouveau vecteur, qui correspond au vecteur unitaire a celui-ci
		Sinon:
		   modifie le vecteur actuel pour le rendre unitaire."""
		norme = self.Norme()
		if new:
			return vector2(self.dx/norme, self.dy/norme)
		self.dx, self.dy = self.dx/norme, self.dy/norme
		self.v = np.array([self.dx, self.dy])
		return 0

	def AngleInDegreeBetween(self, vec):
		"""Renvoie l'angle en degrés entre les deux vecteurs."""
		angle = degrees(acos(self.Scalaire(vec) / (self.Norme() * vec.Norme())))
		return angle

	def AngleInRadiansBetween(self, vec):
		"""Renvoie l'angle en radians entre les deux vecteurs."""
		angle = degrees(acos(self.Scalaire(vec) / (self.Norme() * vec.Norme())))
		return angle
	# Projection du vecteur sur un autre (NouveauVecteur = (AncienVecteur.dx * AutreVecteur.dx + AncienVecteur.dy * AutreVecteur.dy) / distance (pour le rendre unitaire) )
	# Puis renvoyer le nouveau vecteur généré
	def Projection(self, vecteur):
		return vector2((self.Scalaire(vecteur) / vecteur.Norme()**2) * vecteur.dx, (self.Scalaire(vecteur) / vecteur.Norme()**2) * vecteur.dy)

	def __add__(self, vec):
		return vec2(self.dx + vec.dx, self.dy + vec.dy)

	def __sub__(self, vec):
		return vec2(self.dx - vec.dx, self.dy - vec.dy)

	def __mul__(self, vec):
		if type(vec) == vector2:
			return vec2(self.dx * vec.dx, self.dy * vec.dy)	
		elif type(vec) == int or type(vec) == float:
			return vec2(self.dx * vec, self.dy * vec)

	def __truediv__(self, vec):
		if type(vec) == vector2:
			return vec2(self.dx / vec.dx, self.dy / vec.dy)
		elif type(vec) == int or type(vec) == float:
			return vec2(self.dx / vec, self.dy / vec)

	def __neg__(self):
		return vec2(-self.dx, -self.dy)
	
	def __le__(self, vec):
		if type(vec) == vector2:
			if self.Norme() <= vec.Norme():
				return True
			return False
		elif type(vec) == float or type(vec) == int:
			if self.Norme() <= vec:
				return True
			return False

	def __ge__(self, vec):
		if type(vec) == vector2:
			if self.Norme() >= vec.Norme():
				return True
			return False
		elif type(vec) == float or type(vec) == int:
			if self.Norme() >= vec:
				return True
			return False
			
	def __lt__(self, vec):
		if type(vec) == vector2:
			if self.Norme() < vec.Norme():
				return True
			return False
		elif type(vec) == float or type(vec) == int:
			if self.Norme() < vec:
				return True
			return False

	def __gt__(self, vec):
		if type(vec) == vector2:
			if self.Norme() > vec.Norme():
				return True
			return False
		elif type(vec) == float or type(vec) == int:
			if self.Norme() > vec:
				return True
			return False

	def __eq__(self, vec):
		if type(vec) == vector2:
			if self.Norme() == vec.Norme():
				return True
			return False
		elif type(vec) == int or type(vec) == float:
			if self.Norme() == vec:
				return True
			return False
	
	def __ne__(self, vec):
		if type(vec) == vector2:
			if self.Norme() != vec.Norme():
				return True
			return False
		elif type(vec) == int or type(vec) == float:
			if self.Norme() != vec:
				return True
			return False

	def __str__(self):
		return f"({self.dx} {self.dy})"
	
	def __getitem__(self, x):
		if not x in ['dx', 'dy']:
			raise KeyError("la composante demandée n'existe pas. Essayer 'dx', 'dy'.")
		else:
			if x == 'dx': return self.dx
			if x == 'dy': return self.dy

	def __setitem__(self, x, value):
		if not x in ['dx', 'dy']:
			raise KeyError("la composante demandée n'existe pas. Essayer 'dx', 'dy'.")
		else:
			if x == 'dx': self.dx = value
			if x == 'dy': self.dy = value


class Force(vector2):
	def __init__(self, dx, dy):
		super().__init__(dx, dy)

	def Apply(self, point, method): # avec comme methode possible : euler, runge-kutta
		return 0

class Point2d:
    def __init__(self, x, y):
        self.x, self.y = x, y

class vector3:
	"""Crée un nouveau vecteur. Indiquer les x, y, z (=dx, dy, dz)"""
	def __init__(self, dx, dy, dz, w = 1):
		self.dx, self.dy, self.dz = dx, dy, dz
		self.w = w
	
	def Scalaire(self, vec):
		"""Don le rpoduit scalaire des deux vecteurs"""
		return (self.dx * vec.dx) + (self.dy * vec.dy) + (self.dz * vec.dz)
	
	def Norme(self):
		"""Donne la norme du vecteur"""
		return sqrt(self.dx**2 + self.dy**2 + self.dz **2)

	def Unitaire(self):
		"""Rend le vecteur unitaire"""
		norme = self.Norme()
		if norme != 0:
			self.dx /= norme
			self.dy /= norme
			self.dz /= norme

	def CrossProduct(self, vec):
		"""Produit croisé entre les deux vecteurs. Crée un nouveau vecteur hortogonal aux deux autres."""
		Nx = self.dy * vec.dz - self.dz * vec.dy
		Ny = self.dz * vec.dx - self.dx * vec.dz
		Nz = self.dx * vec.dy - self.dy * vec.dx
		return vector3(Nx, Ny, Nz)

	def AngleInDegree(self, vec):
		"""Donne l'angle en degrés entre les deux vecteurs."""
		return degrees(acos( self.Scalaire(vec) / self.Norme() * vec.Norme()))
	
	def AngleInRadian(self, vec):
		"""Donne l'angle en radians entre les deux vecteurs."""
		return acos( self.Scalaire(vec) / self.Norme() * vec.Norme())

	def __add__(self, vec):
		return vec3(self.dx + vec.dx, self.dy + vec.dy, self.dz + vec.dz)

	def __sub__(self, vec):
		return vec3(self.dx - vec.dx, self.dy - vec.dy, self.dz - vec.dz)

	def __mul__(self, vec):
		if type(vec) == vector3:
			return vec3(self.dx * vec.dx, self.dy * vec.dy, self.dz * vec.dz)	
		elif type(vec) == int or type(vec) == float:
			return vec3(self.dx * vec, self.dy * vec, self.dz * vec)

	def __truediv__(self, vec):
		if type(vec) == vector3:
			return vec3(self.dx / vec.dx, self.dy / vec.dy, self.dz / vec.dz)
		elif type(vec) == int or type(vec) == float:
			return vec3(self.dx / vec, self.dy / vec, self.dz / vec)

	def __neg__(self):
		return vec3(-self.dx, -self.dy, -self.dz)
	
	def __le__(self, vec):
		if type(vec) == vector3:
			if self.Norme() <= vec.Norme():
				return True
			return False
		elif type(vec) == float or type(vec) == int:
			if self.Norme() <= vec:
				return True
			return False

	def __ge__(self, vec):
		if type(vec) == vector3:
			if self.Norme() >= vec.Norme():
				return True
			return False
		elif type(vec) == float or type(vec) == int:
			if self.Norme() >= vec:
				return True
			return False

	def __lt__(self, vec):
		if type(vec) == vector3:
			if self.Norme() < vec.Norme():
				return True
			return False
		elif type(vec) == float or type(vec) == int:
			if self.Norme() < vec:
				return True
			return False

	def __gt__(self, vec):
		if type(vec) == vector3:
			if self.Norme() > vec.Norme():
				return True
			return False
		elif type(vec) == float or type(vec) == int:
			if self.Norme() > vec:
				return True
			return False

	def __str__(self):
		return f"({self.dx} {self.dy} {self.dz})"
	
	def __getitem__(self, x):
		if not x in ['dx', 'dy', 'dz']:
			raise KeyError("la composante demandée n'existe pas. Essayer 'dx', 'dy', 'dz'.")
		else:
			if x == 'dx': return self.dx
			if x == 'dy': return self.dy
			if x == 'dz': return self.dz

	def __setitem__(self, x, value):
		if not x in ['dx', 'dy', 'dz']:
			raise KeyError("la composante demandée n'existe pas. Essayer 'dx', 'dy', 'dz'.")
		else:
			if x == 'dx': self.dx = value
			if x == 'dy': self.dy = value
			if x == 'dz': self.dz = value

class Point3d:
	def __init__(self, x, y, z, w = 1):
		self.x, self.y, self.z = x, y, z
		self.w = w

def Point3dFromVector(vector):
		return vector3(vector.dx, vector.dy, vector.dz)

def Vector3dFromPoint(point):
		return vector3(point.x, point.y, point.z)

vec3 = vector3

def PickTheNearest(a, b, Midle):
	if abs(Midle - a) < abs(Midle - b):
		return a
	return b


class Matrice4x4:
	def __init__(self):
		self.arr = [[0 for i in range(4)] for x in range(4)]

	def MultiplieMatricePoint(self, i:Point3d):
		o = Point3d(0, 0, 0)
		o.x = i.x * self.arr[0][0] + i.y * self.arr[1][0] + i.z * self.arr[2][0] + i.w * self.arr[3][0]
		o.y = i.x * self.arr[0][1] + i.y * self.arr[1][1] + i.z * self.arr[2][1] + i.w * self.arr[3][1]
		o.z = i.x * self.arr[0][2] + i.y * self.arr[1][2] + i.z * self.arr[2][2] + i.w * self.arr[3][2]
		o.w = i.x * self.arr[0][3] + i.y * self.arr[1][3] + i.z * self.arr[2][3] + i.w * self.arr[3][3]

		return o

	def MatriceMultiplieVecteur(self, i:vector3):
		o = vec3(0, 0, 0)
		o.dx = i.dx * self.arr[0][0] + i.dy * self.arr[1][0] + i.dz * self.arr[2][0] + i.w * self.arr[3][0]
		o.dy = i.dx * self.arr[0][1] + i.dy * self.arr[1][1] + i.dz * self.arr[2][1] + i.w * self.arr[3][1]
		o.dz = i.dx * self.arr[0][2] + i.dy * self.arr[1][2] + i.dz * self.arr[2][2] + i.w * self.arr[3][2]
		o.w  = i.dx * self.arr[0][3] + i.dy * self.arr[1][3] + i.dz * self.arr[2][3] + i.w * self.arr[3][3]
		return o

	def MatriceMultiplieMatrice(self, m1):
		m2 = self
		o = Matrice4x4()
		for c in range(4):
			for r in range(4):
				o.arr[r][c] = m1.arr[r][0] * m2.arr[0][c] + m1.arr[r][1] * m2.arr[1][c] + m1.arr[r][2] * m2.arr[2][c] + m1.arr[r][3] * m2.arr[3][c]
		return o

	def MatriceIdentite(self):
		self.arr[0][0] = 1
		self.arr[1][1] = 1
		self.arr[2][2] = 1
		self.arr[3][3] = 1

	
	def MatriceInverse(self):
		matrix = Matrice4x4()
		matrix.arr[0][0] = self.arr[0][0]; matrix.arr[0][1] = self.arr[1][0]; matrix.arr[0][2] = self.arr[2][0]; matrix.arr[0][3] = 0.0;
		matrix.arr[1][0] = self.arr[0][1]; matrix.arr[1][1] = self.arr[1][1]; matrix.arr[1][2] = self.arr[2][1]; matrix.arr[1][3] = 0.0;
		matrix.arr[2][0] = self.arr[0][2]; matrix.arr[2][1] = self.arr[1][2]; matrix.arr[2][2] = self.arr[2][2]; matrix.arr[2][3] = 0.0;
		matrix.arr[3][0] = -(self.arr[3][0] * matrix.arr[0][0] + self.arr[3][1] * matrix.arr[1][0] + self.arr[3][2] * matrix.arr[2][0]);
		matrix.arr[3][1] = -(self.arr[3][0] * matrix.arr[0][1] + self.arr[3][1] * matrix.arr[1][1] + self.arr[3][2] * matrix.arr[2][1]);
		matrix.arr[3][2] = -(self.arr[3][0] * matrix.arr[0][2] + self.arr[3][1] * matrix.arr[1][2] + self.arr[3][2] * matrix.arr[2][2]);
		matrix.arr[3][3] = 1.0;
		return matrix

vec2 = vector2
vec3 = vector3