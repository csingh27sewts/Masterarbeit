
###############################
## CLOTH_SEWTS_MINIMAL : 3X3 ##  
###############################

# TYPE 0 : FIXED INITIALIZATION #

EXPERIMENT 0_1 : Move corner to fixed position with action on corner
EXPERIMENT 0_2: Move two adjacent seam points to fixed positions with action on corner

# TYPE 1 : RANDOM INITIALIZATION #

EXPERIMENT 1_1 : Move corner to fixed position (randomly initialized) with action on corner
EXPERIMENT 1_2 : Move two adjacent seam points to fixed positions (randomly initialized) with action on corner
EXPERIMENT 1_3 : Move two adjacent seam points to fixed positions (randomly initialized) with action on one of the adjacent points

# TYPE 2 : PREMANIPULATION # 

EXPERIMENT 2_1 : Move two adjacent seam points to z=0 (randomly initialized) with action on one of the adjacent points

##############################
## CLOTH_SEWTS_INTERMEDIATE : 6X6 ##  
##############################

# TYPE 0 : FIXED INITIALIZATION #

EXPERIMENT 0_1 : Move three adjacent points to fixed positions with action on corner

# TYPE 1 : RANDOM INITIALIZATION #

EXPERIMENT 1_1 : Move three adjacent points to fixed positions (randomly initialized) with action on corner
EXPERIMENT 1_2 : Move three adjacent seam points to fixed positions (randomly initialized) with action on one of the adjacent points
EXPERIMENT 1_3 : Move three adjacent seam points to z=0 (randomly initialized) with action on one of the adjacent points

# TYPE 2 : PREMANIPULATION # 

EXPERIMENT 2_1 : Premanipulation experiment to make three cloth points project on straight line

##############################
## CLOTH_SEWTS_FULL : 9X9 ##  
##############################

# TYPE 0 : FIXED INITIALIZATION #

EXPERIMENT 0_1 : Move five adjacent points to fixed positions with action on corner

# TYPE 1 : RANDOM INITIALIZATION #

EXPERIMENT 1_1 : Move five adjacent points to fixed positions (randomly initialized) with action on corner
EXPERIMENT 1_2 : Move five adjacent seam points to fixed positions (randomly initialized) with action on one of the adjacent points
EXPERIMENT 1_3 : Move five adjacent seam points to z=0 (randomly initialized) with action on one of the adjacent points

# TYPE 2 : PREMANIPULATION # 

EXPERIMENT 2_1 : Premanipulation experiment to make five cloth points project on straight line

