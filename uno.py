import GraphicEngine as ge
import os
def display():
	global x1, y2, pass_
	game_engine.Erase()
	if pass_:
		if x1 < 0:
			pass_=False
		
		x1 -= 1*game_engine.DeltaTime
		y2 -= 1*game_engine.DeltaTime
	elif x1 < 400 and y2 < 400:
		x1 += 1*game_engine.DeltaTime
		y2 += 1*game_engine.DeltaTime
	else:
		pass_ = True
	game_engine.Polygon(card, multiplyer=5, x=200, y=200)
	game_engine.Polygon(symbols["four"], multiplyer=0.125, x=200, y=200, fill="red", outline="black")
	game_engine.Polygon(contour, fill="grey", layer=0, outline="grey")
	return

game_engine = ge.Window(target=display,name="Uno", showfps=False, dt=10)

card = [(0,1), (0, 10), (1, 11), (6, 11), (7, 10), (7, 1), (6, 0), (1, 0)]

symbols = {}
to_load = os.listdir("./")
for i in to_load:
	if ".fig" in i:
		name = i
		symbols[i.replace(".fig", "")] = game_engine.LoadFigure(name)
x1 = 0
y2 = 0
pass_ = False
contour = [(0,0),(0,400),(400,400),(400,0),(0,0),(36, 33), (20, 345), (38, 357), (36, 372), (43, 380), (56, 387), (131, 380), (144, 375), (163, 378), (184, 387), (357, 389), (365, 385), (371, 376), (371, 362), (367, 353), (363, 330), (369, 312), (377, 294), (395, 36), (382, 29), (371, 12), (346, 27), (257, 26), (167, 24), (86, 30), (62, 12), (50, 17), (44, 26),(36, 33)]
game_engine.run()