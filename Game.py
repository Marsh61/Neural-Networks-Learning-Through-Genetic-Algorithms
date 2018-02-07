import pygame
import random
import numpy as np
import sys
from pygame.locals import *
from genetic_algorithm import GeneticAlgorithm
from wall_object import FlappyBirdWall
from player_object import FlappyBirdPlayer

class FlappyBirdGame(object): 
    def __init__(self, human_player,num_of_computers, NN_dims, genetic_algorithm, testing = None, screen_width = 640, screen_height = 480):
        pygame.init()
        #Constants 
        self.START_Y = 30
        self.START_X = 15
        self.GRAVITY = 1 
        self.X_SPEED = 3
        self.MAX_Y_SPEED = 2
        self.JUMP_VELOC = -10


        self.testing = testing
        if (self.testing != None):
            self.testing_epochs = testing[0]
            self.walls_cleared_before_winner = testing[1]
            self.num_players_survived = list()

        self.genetic_algorithm = genetic_algorithm

        #epoch_counter
        self.epoch_counter = 0
        #max 10 
        if (num_of_computers > 10 or num_of_computers < 0):
            num_of_computers = 10
        self.NUM_OF_COMPUTERS = num_of_computers
        #Set up screen
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width,screen_height))
        #Init player colours
        self.colours = list()
        self.init_colours()
        self.colour_counter = 0
        #Init Players 
        self.players = list()
        self.PLAYER_RADIUS = 10
        self.human_player = None
        if (human_player == True):
            self.human_player = self.init_player()

        for i in range(self.NUM_OF_COMPUTERS):
            player = self.init_player(NN_dims)
            self.players.append(player)
        
        #Init Walls
        self.walls = list()
        self.num_walls_past = 0


        #game loop
        self.game_on = True
    
    def init_colours(self):
        self.colours.append(pygame.Color(255,0,0,0))
        self.colours.append(pygame.Color(0,255,0,0))
        self.colours.append(pygame.Color(208,0,255,0))
        self.colours.append(pygame.Color(0,225,255,0))
        self.colours.append(pygame.Color(250,255,0,0))
        self.colours.append(pygame.Color(255,110,0,0))
        self.colours.append(pygame.Color(198,198,198,0))
        self.colours.append(pygame.Color(255,255,255,0))
        self.colours.append(pygame.Color(169, 130, 255))
        self.colours.append(pygame.Color(255, 76, 162))
        self.colours.append(pygame.Color(174, 226, 102))
    
    def init_player(self,NN_dims = None):
        start_y = random.randint(self.START_Y, self.screen_height - self.START_Y)
        player = FlappyBirdPlayer(self.START_X, self.screen_height/2,self.colours[self.colour_counter],self.PLAYER_RADIUS, 
            self.X_SPEED, self.MAX_Y_SPEED, self.screen_height,self.screen_width, NN_dims)
        self.colour_counter += 1
        return player

    def game_loop(self):
        self.game_on = True 
        clock = pygame.time.Clock()
        self.SPAWN_WALL = pygame.USEREVENT+1
        pygame.time.set_timer(self.SPAWN_WALL, int((1000) * 2))
        #init wall
        wall = FlappyBirdWall(self.screen_width,self.X_SPEED,self.screen_height,self.PLAYER_RADIUS)
        self.walls.append(wall)
        while (self.game_on):
            clock.tick(100)
            self.draw_screen()
            self.get_AI_output()
            for event in pygame.event.get():
                self.handle_events(event)
            
            self.update_walls()
            self.update_players()
            self.check_collisions()
    
    def reset_with_new_generation(self):
        self.players = self.genetic_algorithm.choose_next_generation(self.players,self.epoch_counter)
        self.epoch_counter += 1
        if (self.testing != None):
            if (self.epoch_counter >= self.testing_epochs):
                self.game_on = False

        if (self.game_on == True):
            self.num_walls_past  = 0
            #print("GENERATION: " + str(self.epoch_counter))
            self.walls.clear()
            for player in self.players:
                player.alive = True
                player.dist_traveled = 0
                start_y = random.randint(self.START_Y, self.screen_height - self.START_Y)
                #player.y_pos = start_y + player.radius
                player.y_pos = self.screen_height/2
                player.x_dist_from_gap = None
                player.y_dist_from_gap = None
            
            if (self.human_player != None):
                self.human_player.alive = True

    
    def check_collisions(self):
        if (self.human_player != None):
            for i in range(len(self.walls)):
                if (self.human_player.collision_rect.colliderect(self.walls[i].bottom_wall)
                    or self.human_player.collision_rect.colliderect(self.walls[i].top_wall)):
                    # Player Dies Code Here
                    self.human_player.alive = False
                # prevent unesscary collision detection
                if (i > 2):
                    break
        
        player_still_alive = False
        if (self.human_player != None):
            if (self.human_player.alive == True):
                    player_still_alive = True
        
        
        for player in self.players:
            #check for collisions
            for i in range(len(self.walls)):
                if (player.collision_rect.colliderect(self.walls[i].bottom_wall)
                    or player.collision_rect.colliderect(self.walls[i].top_wall)):
                    # Player Dies Code Here
                    player.alive = False
                # prevent unesscary collision detection
                if (i > 2):
                    break

            if (player.alive == True):
                player_still_alive = True

        if (player_still_alive == False and self.game_on == True):
            if (self.testing != None):
                self.num_players_survived.append(0)
            self.reset_with_new_generation()

    def get_AI_output(self):
        for player in self.players:
            if (player.x_dist_from_gap == None or player.y_dist_from_gap == None):
                return
                
            #Feed NN input and retrieve output then jump on output
            output = player.get_prediction()
            #1 means jump, 0 means dont jump 
            if (output == 1):
                player.set_y_velocity(self.JUMP_VELOC)

    def update_players(self):
        if (self.human_player != None):
            self.human_player.update_velocity(0,self.GRAVITY)
            self.human_player.update_pos()
            for wall in self.walls:
                if (self.human_player.x_pos < wall.gap_center[0]):
                    #update the NN features 
                    self.human_player.get_x_y_from_gap(wall)
                    break

        for player in self.players:
            player.update_velocity(0,self.GRAVITY)
            player.update_pos()
            for wall in self.walls:
                if (player.x_pos < wall.gap_center[0]):
                    #update the NN features 
                    player.get_x_y_from_gap(wall)
                    break

    def update_walls(self):            
        #clear out a completed wall
        if (len(self.walls) > 0):
            if (self.walls[0].x_pos < (0-self.walls[0].WALL_WIDTH)):
                self.walls.pop(0)
                self.num_walls_past += 1
                if (self.testing != None):
                    if (self.walls_cleared_before_winner <= self.num_walls_past):
                        self.check_for_winner()

        #update wall position
        for wall in self.walls:
            wall.update_pos()

    def check_for_winner(self):
        num_players_survived = 0
        for player in self.players:
            if (player.alive == True):
                num_players_survived += 1
                player.alive = False
        
        if (self.game_on == True):        
            self.num_players_survived.append(num_players_survived)
            self.reset_with_new_generation()
        
    def handle_events(self,event):
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == KEYDOWN:
            if event.key == pygame.K_SPACE:
                if (self.human_player != None):
                    self.human_player.set_y_velocity(self.JUMP_VELOC)

            if event.key == pygame.K_UP:
                wall = FlappyBirdWall(self.screen_width,self.X_SPEED,self.screen_height,self.PLAYER_RADIUS)
                self.walls.append(wall)

        if event.type == self.SPAWN_WALL:
            wall = FlappyBirdWall(self.screen_width,self.X_SPEED,self.screen_height,self.PLAYER_RADIUS)
            self.walls.append(wall)



    def draw_screen(self):
        self.screen.fill(0)
        if (self.human_player != None):
            self.human_player.draw_self(self.screen)
        for player in self.players:
            player.draw_self(self.screen)
        for wall in self.walls:
            wall.draw_self(self.screen)
        pygame.display.update()



if __name__ == "__main__":
    
    x = input("Input A Number (1-4) to Select your Genetic Algorithm Type\n 1: Simulated Annealing \n 2: Diversity Selection \n 3: Diversity Selection and Simulated Annealing \n 4: Neither Diversity Selection or Simulated Annealing\n")
    if (int(x) == 1):
        #Simulated Annealing
        print("\nRunning: Simulated Annealing")
        print("\nPress up arrow key to spawn a wall (helps kill the current generation)")
        print("Press space to jump and play (optional)")
        ga = GeneticAlgorithm(1,.4,.7,10,.7,.25,85,-1)
        game = FlappyBirdGame(True,10,(2,6,2),ga)
        game.game_loop()

    if (int(x) == 2):
        #Diversity Selection 
        print("\nRunning: Diversity Selection")
        print("\nPress up arrow key to spawn a wall (helps kill the current generation)")
        print("Press space to jump and play (optional)")
        ga = GeneticAlgorithm(1,.4,.7,10,.7,.25,-1,80)
        game = FlappyBirdGame(True,10,(2,6,2),ga)
        game.game_loop()

    if (int(x) == 3):
        #Diversity Selection and Simulated Annealing
        print("\nRunning: Diversity Selection and Simulated Annealing")
        print("\nPress up arrow key to spawn a wall (helps kill the current generation)")
        print("Press space to jump and play (optional)")
        ga = GeneticAlgorithm(1,.4,.7,10,.7,.25,85,80)
        game = FlappyBirdGame(True,10,(2,6,2),ga)
        game.game_loop()

    if (int(x) == 4):
        #Neither Diversity Selection or Simulated Annealing
        print("\nRunning: Neither Diversity Selection or Simulated Annealing")
        print("\nPress up arrow key to spawn a wall (helps kill the current generation)")
        print("Press space to jump and play (optional)")
        ga = GeneticAlgorithm(1,.4,.7,10,.7,.25,-1,-1)
        game = FlappyBirdGame(True,10,(2,6,2),ga)
        game.game_loop()
    