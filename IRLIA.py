import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
import torch.optim as optim
from model import GaussianPolicy, QNetwork, DeterministicPolicy
from autoencoder import autoencoder

import matplotlib.pyplot as plt

l=torch.nn.MSELoss()
ll=torch.nn.PairwiseDistance(p=2,keepdim=True)
class Agent(object):
    def __init__(self, num_inputs, action_space, args):
        self.args=args
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.alpha1=args.alpha1
        
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.l=[]
        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    def update_model1(self,model, new_params):
        index = 0
        for params in model.parameters():
            params_length = len(params.view(-1))
            new_param = new_params[index: index + params_length]
            new_param = new_param.view(params.size())
            params.data.copy_(new_param.to("cuda:0")+params.to("cuda:0"))
            index += params_length
    
    def update_parametersafter(self, memory, batch_size, updates,env,enco):
        '''
        Temporarily updates the parameters of the first agent.

        Parameters
        ----------
        memory : class 'replay_memory.ReplayMemory'
            
        batch_size : int
        
        updates : int
        
        env : 'gym.wrappers.time_limit.TimeLimit'
            The environment of interest
        enco : class
            The corresponding autoencoder
        '''
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            
            cat=torch.cat((next_state_batch,next_state_action),dim=-1)
            s=enco(cat)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            
            next_q_value = reward_batch +self.alpha* ll(cat,s) + mask_batch * self.gamma * (min_qf_next_target)
        
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        policy_loss = (-min_qf_pi).mean()# Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
        self.policy_optim.step()
        
        alpha_loss = torch.tensor(0.).to(self.device)
        alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    def update_parametersdeter(self, memory, batch_size, updates,env,enco):
        '''
        Updates the paratmeters of the second agent.

        Parameters
        ----------
        memory : class 'replay_memory.ReplayMemory'
            
        batch_size : int
        
        updates : int
        
        env : 'gym.wrappers.time_limit.TimeLimit'
            The environment of interest
        enco : class
            The corresponding autoencoder


        '''
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            
            cat=torch.cat((state_batch,action_batch),dim=-1)
            s=enco(cat)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            act,_,_=self.policy.sample(state_batch)
            
            
            next_q_value = reward_batch -self.alpha1* ((ll(cat,s)-torch.min(ll(cat,s)))/(torch.max(ll(cat,s))-torch.min(ll(cat,s)))
                                                      )*ll(act,action_batch) + mask_batch * self.gamma * (min_qf_next_target) #refer to the paper
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
    
        policy_loss = (-min_qf_pi).mean()# J_{π_2} = 𝔼st∼D,εt∼N[− Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
        self.policy_optim.step()
        
        alpha_loss = torch.tensor(0.).to(self.device)
        alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    
    def X (self, ss,a,  memory, batch_size, updates,env,enco, Qdac, pidac, QTdac,args,tenco, normalization=False):
        '''
        Updates the parameters of the first agent (with the influence function and intrinsic rewards).

        Parameters
        ----------
        ss : numpy array
           Current state
            
        a : numpy array
           Action taken in ss
            
        memory : class 'replay_memory.ReplayMemory'
            
        batch_size : int
        
        updates : int
        
        env : 'gym.wrappers.time_limit.TimeLimit'
            The environment of interest
            
        enco : 
            The corresponding autoencoder
            
        Qdac : 
            The critic network of the second agent.

        pidac : 
           The policy network of the second agent.
            
        QTdac : 
           The target critic network of the second agent.
            
        args : 
           Hyperparameters determined by the user.
            
        tenco : 
           A virtual/proxy autoencoder used to calculate the frequency of (ss,a) w.r.t first agent's policy
        '''
        
        Qdac_optim = Adam(Qdac.parameters(), lr=args.lr)
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            
            cat=torch.cat((next_state_batch,next_state_action),dim=-1)
            s=enco(cat)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            
            if normalization:
                next_q_value = reward_batch +self.alpha* ll(cat,s)/torch.max(ll(cat,s)) + mask_batch * self.gamma * (min_qf_next_target)
            else:
                next_q_value = reward_batch +self.alpha* ll(cat,s) + mask_batch * self.gamma * (min_qf_next_target)
        
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        

        
        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        
        
        
        
                
        #Update the proxy Qs of the DAC according to the 
        
        #Qdac, pidac, QTdac
        with torch.no_grad():
            next_state_action, _, _ = pidac.sample(next_state_batch)
            
            qf1_next_target, qf2_next_target = QTdac(next_state_batch, next_state_action)
            
            cat=torch.cat((state_batch,action_batch),dim=-1)
            s=tenco(cat)
            
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            act,_,_=pidac.sample(state_batch)
            
            
            next_q_value = reward_batch -self.alpha1* ((ll(cat,s)-torch.min(ll(cat,s)))/(torch.max(ll(cat,s))-torch.min(ll(cat,s)))
                                                      )*ll(act,action_batch) + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = Qdac(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)
        
        Qdac_optim.zero_grad()                              #Update 2nd Agent's proxy networks 
        qf1_loss.backward()
        Qdac_optim.step()
        
        Qdac_optim.zero_grad()
        qf2_loss.backward()
        Qdac_optim.step()
        
        #Find the value of F--the influence function
        pi_BAC,_,_=self.policy.sample(state_batch)
        
        with torch.no_grad():
            next_state_action, _, _ = pidac.sample(next_state_batch)
            qf1_next_target, qf2_next_target = QTdac(next_state_batch, next_state_action)
            
            cat=torch.cat((state_batch,pi_BAC),dim=-1)
            s=enco(cat)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            act,_,_=pidac.sample(state_batch)
            
            
            next_q_value = reward_batch -self.alpha1* ((ll(cat,s)-torch.min(ll(cat,s)))/(torch.max(ll(cat,s))-torch.min(ll(cat,s)))
                                                      )*ll(act,pi_BAC) + mask_batch * self.gamma * (min_qf_next_target)
        
        qf1, qf2 = Qdac(state_batch, pi_BAC)
        
        
        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value)
        
        minlossinf=torch.min(qf1_loss,qf2_loss)
        qf1_pi, qf2_pi = self.critic(state_batch, pi_BAC)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = (-min_qf_pi).mean()
        
        policy_loss+=0.1*minlossinf      #Regulate the objective function of the first agent by adding F
        self.policy_optim.zero_grad()
        policy_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
        self.policy_optim.step()
        
        
        alpha_loss = torch.tensor(0.).to(self.device)
        alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    # Save model parameters   
    def save_model(self, env_name, enco, suffix="", actor_path=None, critic_path=None, enco_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/actor/IRLIA_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/critic/IRLIA_critic_{}_{}".format(env_name, suffix)
        if enco_path is None:
            enco_path = "models/enco/IRLIA_enco_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(enco.state_dict(), enco_path)
        

    # Load model parameters
    def load_model(self, enco, actor_path, critic_path, enco_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if enco_path is not None:
            enco.load_state_dict(torch.load(enco_path))

