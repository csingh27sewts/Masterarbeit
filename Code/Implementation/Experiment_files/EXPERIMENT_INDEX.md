EXPERIMENT #1 : 
Description - Simplistic experiment to make one cloth point move to a fixed position    
Goal - Move cloth corner G00 to center (0, 0)  
State - (x,y,z) of G00
Reward - 500 when it reaches , 500 - 10 * dist from center upto dist of 0.05, -10 * dist upto dist of 0.1 and -100 * dist otherwise   
