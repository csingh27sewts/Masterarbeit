
# EXPERIMENT #0 : 
*cloth_sewts_minimal.py*
Output - SAC_0  
Description - Simplistic experiment to make one cloth point move to a fixed position    
Cloth - 3X3  
Goal - Move cloth corner G00 to center (0, 0)  
State - (x,y,z) of G00
Reward - 500 when it reaches , 500 - 10 * dist from center upto dist of 0.05, -10 * dist upto dist of 0.1 and -100 * dist otherwise  
 
# EXPERIMENT #1 : 
*cloth_sewts_minimal_1.py* 
Output - SAC_1  
Description - Simplistic experiment to make one cloth (randomly initialized) point move to a fixed position    
Cloth - 3X3  
Goal - Move cloth corner G00 to center (0, 0)  
State - (x,y,z) of G00
Reward - 500 when it reaches , 500 - 10 * dist from center upto dist of 0.05, -10 * dist upto dist of 0.1 and -100 * dist otherwise  
 
 
# EXPERIMENT #1_1 : 
Output - SAC_1_1  
*cloth_sewts_minimal_1_1.py* 
Description - Simplistic experiment to make one cloth point move to a fixed position    
Cloth - 3X3  
Goal - Move cloth corner G00 to center (0, 0)  
State - (x,y,z) of G00
Reward - 500 when it reaches , 500 - 10 * dist from center upto dist of 0.05, -10 * dist upto dist of 0.1 and -100 * dist otherwise  (Normalized reward divided by 500)
 
# EXPERIMENT #2 :
Output - SAC_2   
*cloth_sewts_minimal_2.py* 
Description - Simplistic experiment to make two cloth points move to fixed positions
Cloth - 3X3  
Goal - Move cloth corner G00 to center (0, 0) and G01 to (0,0.03)  
State - (x,y,z) of G00
Reward - 500 when it reaches , 500 - 10 * dist from center upto dist of 0.05, -10 * dist upto dist of 0.1 and -100 * dist otherwise  

# EXPERIMENT #3 : 
Output - SAC_3  
*cloth_sewts_minimal_3.py* 
Description - Simplistic experiment to make three cloth points move to fixed positions
Cloth - 3X3  
Goal - Move cloth corner G00 to center (0, 0) , G01 to (0,0.03) and G02 to (0,0.06)  
State - (x,y,z) of G00
Reward - 500 when it reaches , 500 - 10 * dist from center upto dist of 0.05, -10 * dist upto dist of 0.1 and -100 * dist otherwise  

# EXPERIMENT #4 :
Output - SAC_4   
*cloth_sewts_minimal_4.py* 
Description - Simplistic experiment to make three cloth points move to fixed positions
Cloth - 4X4  
Goal - Move cloth corner G00 to center (0, 0) , G01 to (0,0.03) and G02 to (0,0.06)  
State - (x,y,z) of G00
Reward - 500 when it reaches , 500 - 10 * dist from center upto dist of 0.05, -10 * dist upto dist of 0.1 and -100 * dist otherwise  

# EXPERIMENT #5 : 
Output - SAC_5  
*cloth_sewts_minimal_5.py* 
Description - Simplistic experiment to make three cloth points move to fixed positions
Cloth - 5X5  
Goal - Move cloth corner G00 to center (0, 0) , G01 to (0,0.03) and G02 to (0,0.06)  
State - (x,y,z) of G00
Reward - 500 when it reaches , 500 - 10 * dist from center upto dist of 0.05, -10 * dist upto dist of 0.1 and -100 * dist otherwise  

# EXPERIMENT #6 :  
Output - SAC_6  
*cloth_sewts_minimal_6.py* 
Description - Simplistic experiment to make four cloth points move to fixed positions
Cloth - 6X6  
Goal - Move cloth corner G00 to center (0, 0) , G01 to (0,0.03) and G02 to (0,0.06)  
State - (x,y,z) of G00
Reward - 500 when it reaches , 500 - 10 * dist from center upto dist of 0.05, -10 * dist upto dist of 0.1 and -100 * dist otherwise  

# EXPERIMENT #7 : 
*cloth_sewts_minimal_7.py* 
Description - Simplistic experiment to make four cloth points move to fixed positions
Cloth - 9X9  
Goal - Move cloth corner G00 to center (0, 0) , G01 to (0,0.03) , G02 to (0,0.06) and G03 to (0,0.09)  
State - (x,y,z) of G00
Reward - 500 when it reaches , 500 - 10 * dist from center upto dist of 0.05, -10 * dist upto dist of 0.1 and -100 * dist otherwise  

# EXPERIMENT #8 : 
*cloth_sewts_minimal_8.py* 
Description - Simplistic experiment to make five cloth points move to fixed positions
Cloth - 9X9  
Goal - Move cloth corner G00 to center (0, 0) , G01 to (0,0.03) , G02 to (0,0.06), G03 to (0,0.09) and G04 to (0,0.12)
State - (x,y,z) of G00
Reward - 500 when it reaches , 500 - 10 * dist from center upto dist of 0.05, -10 * dist upto dist of 0.1 and -100 * dist otherwise  

# EXPERIMENT #9 : 
*cloth_sewts_minimal_9.py* 
Description - Simplistic experiment to make five cloth points move to fixed positions
Cloth - 9X9  
Goal - Move cloth corner G00 to center (0, 0) , G01 to (0,0.03) , G02 to (0,0.06), G03 to (0,0.09) and G04 to (0,0.12)
State - (x,y,z) of G00
Reward - 500 when it reaches , 500 - 10 * dist from center upto dist of 0.05, -10 * dist upto dist of 0.1 and -100 * dist otherwise  

# EXPERIMENT #10 : 
*cloth_sewts_minimal_10.py*  
Description - First premanipulation experiment to make five cloth points move to fixed positions
Cloth - 9X9  
Goal - Move cloth corner points G00, G01, G02, G03 to straight line  
State - (x,y,z) of G00  
Reward - 500 when it reaches , 500 - 10 * dist from center upto dist of 0.05, -10 * dist upto dist of 0.1 and -100 * dist otherwise 
 
# EXPERIMENT #11 : 
*cloth_sewts_minimal_11.py*  
Description - First premanipulation experiment to make five cloth points fall in a straight line
Cloth - 9X9  
Goal - Move cloth corner points G00, G01, G02, G03 to straight line  
State - (x,y,z) of G00  
Reward - 500 when it reaches , 500 - 10 * dist from center upto dist of 0.05, -10 * dist upto dist of 0.1 and -100 * dist otherwise 

# EXPERIMENT #12 : 
*cloth_sewts_minimal_12.py*  
Description - First premanipulation experiment to make five cloth points fall in a straight line - action can be any of the corners
Cloth - 9X9  
Goal - Move cloth corner points G00, G01, G02, G03 to straight line  
State - (x,y,z) of G00  
Reward - 500 when it reaches , 500 - 10 * dist from center upto dist of 0.05, -10 * dist upto dist of 0.1 and -100 * dist otherwise 

# EXPERIMENT #13 : 
*cloth_sewts_minimal_13.py*  
Description - First premanipulation experiment to make five cloth points fall in a straight line - action can be any of the 5 points
Cloth - 9X9  
Goal - Move cloth corner points G00, G01, G02, G03 to straight line  
State - (x,y,z) of G00  
Reward - 500 when it reaches , 500 - 10 * dist from center upto dist of 0.05, -10 * dist upto dist of 0.1 and -100 * dist otherwise 

# EXPERIMENT #14 : 
*cloth_sewts_minimal_13.py*  
Description - First premanipulation experiment to make five cloth points (from edge selected based on edge weights) fall in a straight line - action can be any of the 5 points
Cloth - 9X9  
Goal - Move cloth corner points G00, G01, G02, G03 to straight line  
State - (x,y,z) of G00  
Reward - 500 when it reaches , 500 - 10 * dist from center upto dist of 0.05, -10 * dist upto dist of 0.1 and -100 * dist otherwise 


