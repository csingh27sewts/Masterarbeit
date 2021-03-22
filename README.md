Run :  
*Launch the training*
cd Code  
python3 cloth_sac_main.py  
 
NOTE : To test a new experiment run cloth_minimal_main.py. This is a skeleton implementation with no learning  
NOTE : To test general environments like InvertedPendulum with SAC learning, CartPole, etc. use general_sac_main.py  
 
Environment selection :  
*Select the environment to train*  
Code/cloth_sac_main.py  

Environment definition :  
*Define states, actions, rewards for the environment*   
Code/Packages/dm_control/dm_control/suite/*environment_name*.py  

Algorithm :  
*Modify the algorithm implementation (If needed)*  
Code/SAC/sac_torch.py  
