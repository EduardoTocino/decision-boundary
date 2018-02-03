#import libraries
import pygame,sys
import numpy as np
import math

class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 3
        self.hiddenLayerSize = 20
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*(sum(sum(y-self.yHat))**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
        
def drawBoxes(): #draws 7 colored boxes used for input
	i = 0
	for color in colorlist:
		if i<=2:
			pygame.draw.rect(window,color,(20*i,0,20,20))
		else:
			pygame.draw.rect(window,color,(20*(i-3),20,20,20))
		i += 1	
	pygame.draw.rect(window,(255,255,255),(60,0,40,40))
	
def drawFilledBox(colornumber): #draws box with black border (selected), input 1-7 for colors in colorlist
	drawBoxes()
	if colornumber<=2:
		pygame.draw.rect(window,(0,0,0),(20*colornumber,0,20,20))
		pygame.draw.rect(window,colorlist[colornumber],(20*colornumber+2,2,16,16))
	elif colornumber == 7:
		pygame.draw.rect(window,(0,0,0),(60,0,40,40))
		pygame.draw.rect(window,(255,255,255),(62,2,36,36))
	else:
		pygame.draw.rect(window,(0,0,0),(20*(colornumber-3),20,20,20))
		pygame.draw.rect(window,colorlist[colornumber],(20*(colornumber-3)+2,22,16,16))
		
def drawProgressBox(percent): #draws run box with percent progress in green
	drawFilledBox(7)
	pygame.draw.rect(window, (0,255,0), (62,2,int(percent*36),36))

#initialise pygame, neural network
pygame.init()
NN = Neural_Network()
window = pygame.display.set_mode((800,600))

#initialise colours
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
yellow = (255,255,0)
cyan = (0,255,255)
magenta = (255,0,255)
colorlist = [red,green,blue,yellow,cyan,magenta]

poslist = [] #list of data points (input)
datacolorlist = [] #list of data (output)

pointlist = [] #list of all points in screen
for i in range(800):
	for j in range(600):
		pointlist.append([i/800.,j/600.])

drawBoxes()

while True: #main loop
	
	pygame.display.update()
	
	for event in pygame.event.get(): #every mouse click/keyboard
		if event.type == 5: #if left click
			
			mouseX,mouseY = pygame.mouse.get_pos() #get mouse position
			
			if mouseX<=100 and mouseY<=40: # in box space
				if mouseX>=60: #go box selected
					
					drawFilledBox(7)
					pygame.display.update()
					
					X = np.array(poslist) #create training arrays
					y = np.array(datacolorlist)
					
					for i in range(10000): #train the network
						if i%50 == 0: #show progress every 50 iterations
							drawProgressBox(i/10000.)
							pygame.display.update()
						dJdW1, dJdW2 = NN.costFunctionPrime(X,y)
						NN.W1 -= dJdW1
						NN.W2 -= dJdW2
						
					pixelcolors = NN.forward(np.array(pointlist)) #map each pixel on the screen to a color
					
					for j in range(800): #draw each pixel its color
						if j%5 == 0: #every five rows
							for k in range(len(poslist)): #draw all the data points over the background
								pointx = int(800*poslist[k][0]) 
								pointy = int(600*poslist[k][1])
								
								datapointcolor = tuple([num*255 for num in datacolorlist[k]])
								pygame.draw.circle(window,(0,0,0),(pointx,pointy),7) #draw black outline
								pygame.draw.circle(window,datapointcolor,(pointx,pointy),5) #draw data point
								
							drawFilledBox(7)
							pygame.display.update()
							
						for h in range(600): #draw the background
							pixelcolor = 175*pixelcolors[600*j+h]
							pygame.draw.rect(window,(pixelcolor[0],pixelcolor[1],pixelcolor[2]),(j,h,2,2)) #draw the color of the pixel
					
					drawFilledBox(7)
				else: #if color changed
					colornumber = int(math.floor(mouseX/20.)+3*math.floor(mouseY/20.)) #find out what color the mouse clicked on
					drawFilledBox(colornumber)
					color = colorlist[colornumber] #set the current color

			else: #if data point added
				pygame.draw.circle(window,color,(mouseX,mouseY),5) #draw data point
				poslist.append([mouseX/800.,mouseY/600.]) #record data point to be used for training
				datacolorlist.append([num / 255. for num in list(color)])
				
				
		if event.type == 12: #if quit pressed
			pygame.quit() #quit
			sys.exit()
