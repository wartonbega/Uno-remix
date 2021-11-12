import GraphicEngine as ge
import time
name = input("Save file as (.fig) : ")

def task():
	global fig
	win.Erase()
	win.Line((200, 0), (200, 400))
	win.Line((0,200), (400, 200))
	
	mouse_n = win.GetMouse()[1]
	if mouse_n[0] and (mouse_n[1].x, mouse_n[1].y) not in fig:
		win.Pixel(mouse_n[1].x, mouse_n[1].y)
		fig.append((mouse_n[1].x, mouse_n[1].y))
	for i in range(len(fig)):
		win.Pixel(fig[i][0], fig[i][1])
		if i != len(fig)-1:
			win.Line((fig[i][0], fig[i][1]), (fig[i+1][0], fig[i+1][1]))
	if win.GetKey("q"):
		return False
	elif win.GetKey("r"):
		fig.pop()
	return

win = ge.Window(task, showfps=False, dt=100, name="Figures creator for WB graphic engine")

fig = []
win.run()
time.sleep(1)
if ".fig" in name:
	with open(name, "w") as f:
		for i in fig:
			f.write(str(i[0])+" "+str(i[1])+"\n")
	
else:
	with open(name+".fig", "w") as f:
		for i in fig:
			f.write(str(i[0])+" "+str(i[1])+"\n")
