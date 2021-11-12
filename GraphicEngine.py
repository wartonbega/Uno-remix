from tkinter import *
import time
from PIL import Image
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

	def __getitem__(self, x):
		if x == 0:
			return self.x
		elif x == 1:
			return self.y
		raise IndexError("Il n'existe que deux composantes pour un point : x (=0) et y (=1)")


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

	def Unitaire(self) -> None :
		"""Rend le vecteur unitaire
		Ne renvoie rien
		"""
		norme = self.Norme()
		if norme != 0:
			self.dx /= norme
			self.dy /= norme
			self.dz /= norme

	def __xor__(self):
		"""Produit croisé entre les deux vecteurs. Crée un nouveau vecteur hortogonal aux deux autres."""
		Nx = self.dy * vec.dz - self.dz * vec.dy
		Ny = self.dz * vec.dx - self.dx * vec.dz
		Nz = self.dx * vec.dy - self.dy * vec.dx
		return vector3(Nx, Ny, Nz)

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

def scale(Quad):
    factors = np.linalg.solve(Quad[1:4,:].T, Quad[0,:])
    factors[2] = -factors[2]
    factors = np.concatenate((np.array([1]), factors))   
    return np.diag(factors)

def proj_matrix(Square, Quad):
    Scale = scale(Quad)
    U = (Square.T).dot(Square)
    U = np.linalg.inv(U)
    U = U.dot(Square.T)
    U = U.dot(Scale)
    U = U.dot(Quad)
    return U, np.linalg.inv(Scale)

def proj_map(x, Matrix, Scale):
    y = Scale.dot(x.dot(Matrix))
    return y

vec2 = vector2
vec3 = vector3


class FileFormatError(Exception):
	pass

class Window:
	"""
	Indiquez une cible (=target), une fonction qui vas mettre a jour la fenêtre et un type d'application (si None, alors à vous les commandes !)
	"""
	def __init__(self, target, name="WB Graphic Engine", EngineType = None, showfps=True, color = 'black', dt = 1, width = 400, height = 400):
		self.keysHeld = {}
		self._mouseInformations = {1:(False, None)}
		self.color = color
		self.showFPS = showfps
		self.fps = []
		self.textures = {}

		self.RUN = False

		self.layers = [{}, {}, {}]
		self.objects = 0

		self.target = target
		self.DeltaTime = dt

		self.time1, self.time2 = 0, time.time()
		self.elapsedTime = 1
		self.window = Tk()
		self.appName = name
		self.window.title(name)
		self._canvas = Canvas(self.window, width = width, height = height, bg = self.color)
		self.width, self.height = width, height
		self._canvas.pack()
		self.window.bind('<KeyPress>', self.UserInput)
		self.window.bind('<KeyRelease>', self.UserRelease)
		self.window.bind("<Button>", self.MouseInput)
		self.window.bind("<ButtonRelease>", self.MouseInput)
		self.window.protocol('WM_DELETE_WINDOW', self.Stop)
		self.window.after(1000, self.Loop)

	def run(self):
		self.RUN = True
		self.window.mainloop()

	def Stop(self):
		self.window.destroy()
		self.RUN = False

	def GetKey(self, key):
		if key in self.keysHeld and self.keysHeld[key]:
			return True
		return False
	
	def MouseInput(self, event):
		if "ButtonPress" in str(event):
			self._mouseInformations[event.num] = (True, event)
		if "ButtonRelease" in str(event):
			self._mouseInformations[event.num] = (False, event)

	def GetMouse(self):
		return self._mouseInformations

	def UserInput(self, event):
		self.keysHeld[event.keysym] = True

	def UserRelease(self, event):
		self.keysHeld[event.keysym] = False

	def Loop(self):
		self.time1 = time.time()
		self.elapsedTime = self.time1 - self.time2
		self.time2 = self.time1
		cont = self.target()
		for layer in reversed(range(len(self.layers))):
			for back in self.layers[layer]:
				if "PIX" in back :self._drawPixel(self.layers[layer][back][0], self.layers[layer][back][1], self.layers[layer][back][2])
				elif "LIN" in back:self._drawLine(self.layers[layer][back][0], self.layers[layer][back][1], self.layers[layer][back][2])
				elif "POL" in back:self._drawPolygon(self.layers[layer][back][0], self.layers[layer][back][1], self.layers[layer][back][2], self.layers[layer][back][3], self.layers[layer][back][4])
				elif "TRI" in back:self._drawTriangle(self.layers[layer][back][0], self.layers[layer][back][1], self.layers[layer][back][2], self.layers[layer][back][3], self.layers[layer][back][4], self.layers[layer][back][5])
				elif "TEX" in back:self.drawTexture(self.layers[layer][back][0], self.layers[layer][back][1], self.layers[layer][back][2])
				elif "QAD" in back:self._drawQuad(self.layers[layer][back][0], self.layers[layer][back][1], self.layers[layer][back][2], self.layers[layer][back][3], self.layers[layer][back][4], self.layers[layer][back][5])
				elif "TEQ" in back:self.drawTextureQuad(self.layers[layer][back][0], self.layers[layer][back][1], self.layers[layer][back][2], self.layers[layer][back][3], self.layers[layer][back][4])

		
		self.objects = 0
		self.ShowFps()
		if not cont:
			self.window.after(self.DeltaTime, self.Loop)
		else:
			self.window.destroy()
			
	def ShowFps(self):
		if len(self.fps) >= 1000:
			self.fps.pop(0)
		self.fps.append(1/self.elapsedTime)
		time_text = str(sum(self.fps)/len(self.fps)).split('.')
		if self.showFPS:
			if self.color == "white":
				self._canvas.create_text(40, 10, text = f"{time_text[0]} fps", fill = "black")
			else:
				self._canvas.create_text(40, 10, text = f"{time_text[0]} fps", fill = "white")
				
		self.window.title(self.appName+" "+str(time_text[0])+" fps")

	def Erase(self, object = None):
		if object != None:
			if type(object) == list:
				for obj in object:
					self._canvas.delete(obj)
		else:
			self.layers = [{}, {}, {}]
			self._canvas.delete('all')

	def Pixel(self, x, y, color = 'white', layer = 1):
		self.objects += 1
		if color == 'Transparent' or not (0 < x < self.width and 0 < y < self.height):
			return
		else:
			self.layers[layer][f"PIX{self.objects}"] = (x, y, color)

	def Line(self, start : tuple, end : tuple, color = 'white', layer=1):
		self.objects += 1
		self.layers[layer][f"LIN{self.objects}"] = (start, end, color)

	def Polygon(self, points : list, fill = 'white', layer=1, multiplyer=1, x=0, y=0, outline='white'):
		"""Crée une figure à partir d'une liste de points
		Si le multiple est différent de 1, alors la figure sera redimentionné en fonction
		La figure sera déssinnée en (0,0) par défaut.
		La ligne de contour est désactivée par défaut"""
		self.objects += 1
		self.layers[layer][f"POL{self.objects}"] = (points, fill, multiplyer, x, y)
		if outline != fill:
			for i in range(len(points)-1):
				self.Line((points[i][0]*multiplyer+x, points[i][1]*multiplyer+y), (points[i+1][0]*multiplyer+x, points[i+1][1]*multiplyer+y), outline, layer)
			self.Line((points[0][0]*multiplyer+x, points[0][1]*multiplyer+y), (points[-1][0]*multiplyer+x, points[-1][1]*multiplyer+y), outline, layer)

	def Triangle(self, a, b, c, colorLine = 'black', fill = 'white', draw_lines = True, layer=1):
		self.objects += 1
		self.layers[layer][f"TRI{self.objects}"] = (a, b, c, colorLine, fill, draw_lines)

	def Texture(self, x, y, texture, layer=1):
		self.objects += 1
		self.layers[layer][f"TEX{self.objects}"] = (x, y, texture)
	
	def TextureQuad(self, point1, point2, point3, point4, texture, layer=1):
		self.objects += 1
		self.layers[layer][f"TEQ{self.objects}"] = (point1, point2, point3, point4, texture)
	
	def Quad(self, coor1, coor2, coor3, coor4, fill = 'white', layer=1):
		self.objects += 1
		self.layers[layer][f"QAD{self.objects}"] = (coor1, coor2, coor3, coor4, fill, fill)
		
	def _drawQuad(self, coor1, coor2, coor3, coor4, fill, outline):
		self._canvas.create_polygon(coor1, coor2, coor3, coor4, fill=fill, outline=outline)

	def _drawPixel(self, x, y, color):
		if color == 'Transparent' or not (0 < x < self.width and 0 < y < self.height):
			return
		else:
			self._canvas.create_rectangle(x, y, x+1, y+1, fill=color, outline=color)

	def _drawLine(self, start, end, color):
		self._canvas.create_line(start, end, fill=color)

	def _drawPolygon(self, points, fill, multiplyer, x, y):
		p = points[:]
		for i in range(len(p)):
			p[i] = ((p[i][0] * multiplyer)+x, (p[i][1] * multiplyer)+y)
		self._canvas.create_polygon(p, fill = fill, outline=fill)

	def _drawTriangle(self, a, b, c, colorLine, fill, draw_lines):
		if draw_lines:
			self._drawLine(a, b, color = colorLine)
			self._drawLine(b, c, color = colorLine)
			self._drawLine(c, a, color = colorLine)
		self._drawPolygon([a,b,c], fill = fill)

	def HexFromRgb(self, rgb):
		if len(rgb) == 4:
			if rgb[3] == 0:
				return 'Transparent'
			else:
				rgb = (rgb[0],rgb[1],rgb[2])
		return "#%02x%02x%02x" % rgb

	def drawTexture(self, x, y, texture):
		if texture == None:
			return
		for i in range(len(texture.pixels)):
			for j in range(len(texture.pixels[i])):
				if len(texture.pixels[i][j]) == 4:
					texture.pixels[i][j] = (texture.pixels[i][j][0], texture.pixels[i][j][1], texture.pixels[i][j][2], PickTheNearest(0, 1, texture.pixels[i][j][3]))
				self._drawPixel(j+x, i+y, self.HexFromRgb(texture.pixels[i][j]))

	def drawTextureQuad(self, point1, point2, point3, point4, texture):
		P = np.array([[0,0,1],
					[1,0,1],
					[0,1,1],
					[1,2,1]])
		
		if texture == None:
			return
		for i in range(len(texture.pixels)):
			for j in range(len(texture.pixels[i])):
				if len(texture.pixels[i][j]) == 4:
					texture.pixels[i][j] = (texture.pixels[i][j][0], texture.pixels[i][j][1], texture.pixels[i][j][2], PickTheNearest(0, 1, texture.pixels[i][j][3]))
				E = np.array([[[j,i,1],
							[j+1,i,1],
							[j+1,i+1,1],
							[j,i+1,1]]])
				U, D = proj_matrix(E, P)
				point = proj_map(E, U, D)
				self._drawQuad((point[0][0],point[0][1]) , (point[1][0],point[1][1]) , (point[2][0],point[2][1]) , (point[3][0],point[3][1]) , self.HexFromRgb(texture.pixels[i][j]), self.HexFromRgb(texture.pixels[i][j]))

	def LoadTexture(self, path):
		image = Image.open(path)
		pixels = list(image.getdata())
		width, height = image.size
		pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
		return Texture(pixels)
	
	def LoadTextureMap(self, path, size):
		image = Image.open(path)
		pixels = list(image.getdata())
		width, height = image.size
		pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
		
		textures = {}
		rows = []
		for x in range(height//size):
			for y in range(width//size):
				textures[f"TEX:{x}{y}"] = []
				for z in range(size):
					textures[f"TEX:{x}{y}"].append(pixels[x*size+z][y*size:y*size+size])
		for i in textures:
			textures[i] = Texture(textures[i])
		self.textures = textures
		return textures

	def LoadFigure(self, path):
		if ".fig" in path:
			f = open(path, "r")
			content = f.read()
			f.close()
		else:
			print("Error, need to be end with '.fig'")
			exit(1)
		fig = []
		for i in content.split("\n"):
			if i != "":
				coor = i.split(" ")
				fig.append((int(coor[0]), int(coor[1])))
		return fig

class Texture:
	def __init__(self, pixels, points = [None, None, None, None]):
		"""
		1––––––2   1––––––2
		|@@@@@@| ou |@@@@@@|
		|@@@@@@|     |@@@@@@|
		3––––––4      3––––––4
		"""
		self.pixels = pixels
		self.size = (len(pixels), len(pixels[0]))
		self.point1,self.point2, self.point3, self.point4 = points[0], points[1], points[2], points[3]

