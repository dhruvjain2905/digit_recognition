import pandas as pd
from tensorflow import keras
import pygame
import numpy as np
import matplotlib.pyplot as plt 
import random


clf = keras.models.load_model("mnist_cnn.h5")


class Cube:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.color = (0,0,0)

    def change_color(self):
        self.color = (255,255,255)

    def reset(self):
        self.color = (0,0,0)

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, 30,30))





def get_square(xcor, ycor, a_x, a_y):
    new_all_x = list(a_x)
    new_all_y = list(a_y)
    while xcor%30 != 0:
        xcor -= 1
    final_ind = new_all_x.index(xcor)
    while ycor%30 != 0:
        ycor -= 1
    final_ind = final_ind + (new_all_y.index(ycor)*28)

    return final_ind


def main():
    all_x = np.arange(0,840, 30)
    all_y = np.arange(0,840, 30)

    p = np.zeros(784)

    all_cubes = []

    screen = pygame.display.set_mode((840,840))



    for i in range(len(all_y)):
            for z in range(len(all_x)):
                cube = Cube(all_x[z], all_y[i])
                all_cubes.append(cube)


    while True:


        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                #gh = p.reshape((28,28))
                #plt.imshow(gh, cmap="gray")
                #plt.show()
                quit()

            elif pygame.mouse.get_pressed()[0]:
                mousex, mousey = pygame.mouse.get_pos()
                nec = get_square(mousex, mousey, all_x, all_y)
                p[nec] = 255
                try:
                    if p[nec+28] != 255:
                        p[nec+28] = random.randint(180,255)
                    if p[nec-28] != 255:
                        p[nec-28] = random.randint(180,255)
                    if p[nec+1] != 255:
                        p[nec+1] = random.randint(180,255)
                    if p[nec-1] != 255:
                        p[nec-1] = random.randint(180,255)
                
                except:
                    pass

                u = all_cubes[nec]
                u.color = (255,255,255)
                all_cubes[nec] = u

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    et = np.argmax(clf.predict(p.reshape((1,28,28,1))))

                    print("The prediction is "+str(et))

                if event.key == pygame.K_c:
                    main()
                    




        screen.fill((0,0,0))

        for i in all_cubes:
            i.draw(screen)


        

        pygame.display.update()

main()


