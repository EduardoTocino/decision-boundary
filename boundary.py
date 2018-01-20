import pygame,sys
import numpy as np
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
pygame.init()
NN = Neural_Network()
window = pygame.display.set_mode((800,600))
red = (255,0,0)
blue = (0,0,255)
color = red
poslist = []
colorlist = []
pointlist = []
for i in range(800):
	for j in range(600):
		pointlist.append([i/800.,j/600.])
pygame.draw.rect(window,red,(0,1,20,20))
pygame.draw.rect(window,blue,(20,0,20,20))
pygame.draw.rect(window,(0,255,0),(40,0,20,20))
while True:
	pygame.display.update()
	for event in pygame.event.get():
		if event.type == 5:
			mouseX,mouseY = pygame.mouse.get_pos()
			if mouseX<=20 and mouseY<=20:
				color = red
			elif mouseX<= 40 and mouseY<=20:
				color = blue
			elif mouseX<=60 and mouseY<=20:
				X = np.array(poslist)
				y = np.array(colorlist)
				for i in range(10000):
					dJdW1, dJdW2 = NN.costFunctionPrime(X,y)
					NN.W1 -= dJdW1
					NN.W2 -= dJdW2
				pointcolors = NN.forward(np.array(pointlist))
				for i in range(800):
					if i%3 == 0:
						pygame.display.update()
					for j in range(600):
						pointcolor = 128*pointcolors[600*i+j]
						pygame.draw.rect(window,(pointcolor[0],pointcolor[1],pointcolor[2]),(i,j,2,2))
				for i in range(len(poslist)):
					pointx = int(800*poslist[i][0])
					pointy = int(600*poslist[i][1])
					if colorlist[i] == [1,0,0]:
						colour = red
					else:
						colour = blue
					pygame.draw.circle(window,colour,(pointx,pointy),5)
				pygame.draw.rect(window,red,(0,1,20,20))
				pygame.draw.rect(window,blue,(20,0,20,20))
				pygame.draw.rect(window,(0,255,0),(40,0,20,20))
			else:
				pygame.draw.circle(window,color,(mouseX,mouseY),5)
				poslist.append([mouseX/800.,mouseY/600.])
				if color == red:
					colorlist.append([1,0,0])
				else:
					colorlist.append([0,0,1])
		if event.type == 12:
			pygame.quit()
			sys.exit()
