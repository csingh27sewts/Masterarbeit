Run :  
*Launch the training*
cd Code  
python3 cloth_sac_main.py  
  
Environment selection :  
*Select the environment to train*  
Code/cloth_sac_main.py  

Environment definition :  
*Define states, actions, rewards for the environment*   
Code/Packages/dm_control/dm_control/suite/*environment_name*.py  

Algorithm :  
*Modify the algorithm implementation (If needed)*  
Code/SAC/sac_torch.py  
