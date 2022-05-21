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

Algorithm Explanation :  
1.Define states, actions, rewards  
2. Take steps till 256 steps  
3. Add state, next_state, action, reward to the ReplyBuffer  
4. Learning begins once the ReplyBuffer is filled for one batch size, i.e. 256 steps  
5. Feed the state batch (picked in random order from ReplayBuffer) to Value Network  
6. Feed the next state batch (picked in random order from ReplayBuffer) to Target Value Network  
7. Get actions and log probabilities by feeding state to Actor Network  
8. Feed state and actions obtained from the Actor Network as state-action pairs to Critic Networks  
9. Q Value is taken as min of the two Critic Network outputs  
10. Feed the sampled state again to the Actor Network this time with Reparametrization as True  
11. Obtain actions and log probabilities  
12. Feed state and actions obtained from the Actor Network as state-action pairs to Critic Networks  
13. Feed state and actions obtained from the Actor Network as state-action pairs to Critic Networks  
14. Q Value is taken as min of the two Critic Network outputs  
