INCLUDED FILES
---------------------
 - Graphs Folder
 - activation_functions.py
 - Game.py
 - Genetic_algorithm.py
 - neural_network.py
 - player_object.py
 - wall_object.py
 - ReadMe.txt
 - Neural Networks Learning Through Genetic Algorithms.pdf

DEPENDENCIES
---------------------
 - Python (Tested on version 3.6.2)
 - pygame (Tested on version 1.9.3)

INSTALLATION
---------------------
 - pip install pygame

HOW TO RUN 
---------------------
 - Command is "python Game.py"

you should be given options (1-4) to choose which type of genetic algorithm you would like 
the program to run with (Simulated Annealing, Diversity Selection, Diversity Selection and Simulated Annealing, 
and Neither Diversity Selection or Simulated Annealing) the difference between the genetic algorithm types are 
explained in the "Neural Networks Learning Through Genetic Algorithms.pdf" document.

INSTRUCTIONS
---------------------
Press up arrow key to spawn a wall (helps kill the current generation)
Press space to jump and play (optional)

VALIDATION
---------------------
Observation - When ran the over a short number of cycles the Neural Networks will ethier 
get stuck at every wall and fall into a local maximum (as explained in the PDF document) 
or they will how to play. 

A method I used to make sure that the genetic algorithms were working properly is when 1 
bird makes it past the first wall, press the up arrow key multiple times. Doing this action
will spawn multiple walls and force the bird to die, the next generation should perform 
better than the last.

EXTERNAL LIBARIES
---------------------
Shinners, P. Dudfield, R. VonAppen, M. Pendleton, B. pygame.
http://www.pygame.org

TROUBLESHOOTING
---------------------
NOTE: the game may run slowly on some computers, I believe this is because the game tries
to keep up with a 100 FPS framerate. If you try on a more powerful CPU the game may run 
at its intentional speed.

Everything else should be in working order, however, if things go wrong try the following:

      1) Edit Game.py
      2) remove all lines after line(249) "if __name__ == "__main__":"
      3) insert the following 3 lines below

        ga = GeneticAlgorithm(1,.4,.7,10,.7,.25,85,-1)
        game = FlappyBirdGame(True,10,(2,6,2),ga)
        game.game_loop()

      4) save and run "python Game.py" this should instantly run Simulated Annealing