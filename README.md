## PURPOSE
The purpose of this program is to show how feedforward neural networks can learn through genetic algorithms to solve a simple problem. In the paper ("Neural Networks Learning Through Genetic Algorithms.pdf") I discuss how genetic algorithms "teach" neural networks the solution to a problem, the problems of using genetic algorithms, and how different genetic algorithm techniques compare to one another.

## HOW TO RUN 
 - Command is "python Game.py"

you should be given options (1-4) to choose which type of genetic algorithm you would like 
the program to run with (Simulated Annealing, Diversity Selection, Diversity Selection and Simulated Annealing, 
and Neither Diversity Selection or Simulated Annealing) the difference between the genetic algorithm types are 
explained in the "Neural Networks Learning Through Genetic Algorithms.pdf" document.

## INSTRUCTIONS
Press up arrow key to spawn a wall (helps kill the current generation)

Press space to jump and play (optional)

## VALIDATION
Observation - When ran the over a short number of cycles the Neural Networks will ethier 
get stuck at every wall and fall into a local maximum (as explained in the PDF document) 
or they will how to play. 

A method I used to make sure that the genetic algorithms were working properly is when 1 
bird makes it past the first wall, press the up arrow key multiple times. Doing this action
will spawn multiple walls and force the bird to die, the next generation should perform better than the last.

## DEPENDENCIES
 - Python (Tested on version 3.6.2)
 - pygame (Tested on version 1.9.3)

## INSTALLATION
 - pip install pygame

## EXTERNAL LIBARIES
Shinners, P. Dudfield, R. VonAppen, M. Pendleton, B. pygame.
http://www.pygame.org

## TROUBLESHOOTING
NOTE: the game may run slowly on some computers, I believe this is because the game tries
to keep up with a 100 FPS framerate. If you try on a more powerful CPU the game may run 
at its intentional speed.
