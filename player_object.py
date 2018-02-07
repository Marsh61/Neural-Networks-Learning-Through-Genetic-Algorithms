import pygame
import numpy as np
from neural_network import NeuralNetwork

class FlappyBirdPlayer(object):
    def __init__(self, init_x, init_y, colour, radius, x_speed, max_y_speed, max_height, max_width, network_dims = None, neural_network = None):
        self.x_pos = init_x + radius
        self.y_pos = init_y + radius
        self.colour = colour
        self.radius = radius
        self.x_speed = x_speed
        self.MAX_Y_SPEED = max_y_speed
        self.square_dimensions = 0.0
        self.velocity = (0,0)
        self.MAX_WIDTH = max_width
        self.MAX_HEIGHT = max_height-radius
        self.MIN_HEIGHT = 0 + radius
        self.collision_rect = self.create_collision_rect()
        self.diversity = 0
        self.fitness_rank = 0
        self.alive = True
        self.touching_bottom_or_top = False

        #Neural Network Feature Info
        self.x_dist_from_gap = None
        self.y_dist_from_gap = None
        self.neural_network = neural_network
        if (network_dims != None ):
            self.neural_network = NeuralNetwork(np.asarray(network_dims),0,0)
        
        #winning feature
        self.dist_traveled = 0

    def create_collision_rect(self):
        #math to match circle to square
        diameter = self.radius * 2
        diameter_squared = np.power(diameter,2)
        div_2 = diameter_squared / 2
        final = np.sqrt(div_2)
        self.square_dimensions = final
        #create the collision Rect to match the circle. plus one is padding
        return_rect = pygame.Rect(self.x_pos - (final/2) + 1, self.y_pos - (final/2) + 1, final,final)
        return return_rect

    def update_pos(self):
        if (self.alive == True):
            self.x_pos = self.x_pos + self.velocity[0]
            self.y_pos = self.y_pos + self.velocity[1] 
            if (self.y_pos > self.MAX_HEIGHT):
                self.y_pos = self.MAX_HEIGHT
            if (self.y_pos < self.MIN_HEIGHT):
                self.y_pos = self.MIN_HEIGHT
            self.collision_rect = pygame.Rect(self.x_pos - (self.square_dimensions/2) + 1, 
                self.y_pos - (self.square_dimensions/2) + 1, self.square_dimensions,self.square_dimensions)
            
            if (self.y_pos == 0 + self.radius or self.y_pos == self.MAX_HEIGHT - self.radius):
                self.touching_bottom_or_top = True
            else:
                self.touching_bottom_or_top = False

            if (self.touching_bottom_or_top == False and self.x_dist_from_gap != None and self.y_dist_from_gap != None):
                self.dist_traveled += self.x_speed
            else:
                self.dist_traveled -= 0

    def update_velocity(self, x, y):
        if (self.alive == True):
            changed_velocity = self.velocity[1] + y
            if (changed_velocity > self.MAX_Y_SPEED):
                changed_velocity = self.MAX_Y_SPEED

            self.velocity = (self.velocity[0] + x, changed_velocity)
    
    def set_y_velocity(self, y):
        if (self.alive == True):
            self.velocity = (0, y)
    
    def draw_self(self,display):
        if (self.alive == True):
            pygame.draw.circle(display,self.colour,(int(self.x_pos),int(self.y_pos)), self.radius, 0)
            #pygame.draw.rect(display,pygame.Color(0,255,0),self.collision_rect)
    
    def get_x_y_from_gap(self,flappyBirdWall):
        if (self.alive == True):
            self.x_dist_from_gap = (self.x_pos - flappyBirdWall.gap_center[0]) 
            self.y_dist_from_gap = (self.y_pos - flappyBirdWall.gap_center[1] ) 
            
    
    def get_dist_from_gap(self):
        return np.sqrt(np.power(self.x_dist_from_gap,2) + np.power(self.y_dist_from_gap,2))
    
    def get_fitness(self):
        return self.dist_traveled - self.get_dist_from_gap()

    def get_prediction(self):
        if (self.alive == True):
            if (self.x_dist_from_gap == None or self.y_dist_from_gap == None):
                return 0 
            
            NN_input = np.asmatrix((self.x_dist_from_gap,self.y_dist_from_gap)).T

            output = self.neural_network.test_no_label(NN_input)

            return output
        else:
            return 0
