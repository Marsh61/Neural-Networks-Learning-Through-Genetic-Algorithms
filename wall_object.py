import pygame
import random

class FlappyBirdWall(object):
    def __init__(self, init_x, x_speed, screen_height, player_radius):
        #CONSTANTS
        self.WALL_WIDTH = player_radius
        self.X_SPEED = x_speed
        self.SCREEN_HEIGHT = screen_height
        self.x_pos = init_x
        self.gap_size = (player_radius * 2) * 4
        self.top_wall = 0
        self.top_wall_ends = 0
        self.bottom_wall = 0
        self.bottom_wall_ends = 0
        self.bottom_y = 0
        self.gap_center = 0
        self.create_walls()


    def create_walls(self):
        self.bottom_wall_ends = random.randint(1 + self.gap_size, self.SCREEN_HEIGHT - 1)
        self.bottom_y = self.bottom_wall_ends
        self.bottom_wall_ends = (self.SCREEN_HEIGHT - self.bottom_wall_ends)
        self.bottom_wall = pygame.Rect(self.x_pos,self.bottom_y,self.WALL_WIDTH,  self.bottom_wall_ends)
        self.top_wall_ends = (self.SCREEN_HEIGHT - self.bottom_wall_ends) - self.gap_size
        self.top_wall = pygame.Rect(self.x_pos,0,self.WALL_WIDTH,self.top_wall_ends)
    
    def update_pos(self):
        self.x_pos = self.x_pos - self.X_SPEED
        self.bottom_wall = pygame.Rect(self.x_pos,self.bottom_y,self.WALL_WIDTH,  self.bottom_wall_ends)
        self.top_wall = pygame.Rect(self.x_pos,0,self.WALL_WIDTH,self.top_wall_ends)
        self.gap_center = (self.x_pos,(self.bottom_y - (self.gap_size / 2)))

    def draw_self(self,display):
        pygame.draw.rect(display,pygame.Color(0,0,255),self.top_wall)
        pygame.draw.rect(display,pygame.Color(0,0,255),self.bottom_wall)