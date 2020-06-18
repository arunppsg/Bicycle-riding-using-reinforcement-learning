#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import torch

class Bicycle:

    def __init__(self, **kwargs):

        self.noise = kwargs.setdefault('noise', 0.04)
        
        self.state = np.zeros(5)   #omega, omega_dot, omega_ddot, theta, theta_dot
        #omega is the angle of bicycle to the verticle. 
        #theta is the angle of bicycle w.r.t straight forward. It is the angle of handle bar.
        self.position = np.zeros(5)   #x_f, x_b, y_f, y_b, psi
        self.end = False  #denotes whether end of episode is reached or not
        
        #If omega exceeds +/- 12 degrees, the bicycle falls.
        self.omega_range = np.array([-np.pi * 12 / 180, np.pi * 12 / 180]) #12 degree in SI units.
        self.theta_range = np.array([-np.pi / 2, np.pi / 2]) 
        self.psi_range = np.array([-np.pi, np.pi]) #Not sure why. Reference: https://github.com/amarack/python-rl
        
        #Defining rewards
        self.reward_goal = 0.01
        self.reward_shaping = 0.00004
        self.reward_fall = -1
        
        #Units in meters and kilograms
        self.c = 0.66      #Horizontal distance between point where front wheel touches ground and centre of mass
        self.d_cm = 0.3    #Vertical distance between center of mass and cyclist
        self.h = 0.94      #Height of center of mass over the ground
        self.l = 1.11      #Distance between front tire and back tire at the point where they touch the ground.
        self.m_c = 15.0    #mass of bicycle
        self.m_d = 1.7     #mass of tire
        self.m_p = 60      #mass of cyclist
        self.r = 0.34      #radius of tire
        self.v = 10.0/3.6  #velocity of the bicycle in m / s
        
        #Useful Precomputations
        self.m = self.m_c + self.m_p
        self.inertia_bc = (13. / 3) * self.m_c + self.h ** 2 + self.m_p * (self.h + self.d_cm) ** 2 #inertia of bicycle and cyclist
        self.inertia_dv = (3. / 2) * (self.m_d * self.r ** 2)   #Various inertia of tires
        self.inertia_dl = .5 * (self.m_d * self.r ** 2)    #Various inertia of tires
        self.inertia_dc = self.m_d * self.r ** 2        #Various inertia of tires
        self.sigma_dot = self.v / self.r
        
        #Simulation constants
        self.gravity = 9.8
        self.delta_time = 0.02
        self.sim_steps = 10
        
        self.goal_loc = np.array([1000., 0])
        self.goal_rsqrd = 100.0 # Square of the radius around the goal (10m)^2
        
    def reset(self):
        self.state.fill(0.0)
        self.position.fill(0.0)
        self.position[3] = self.l
        self.position[4] = np.arctan((self.position[1]-self.position[0])/(self.position[2] - self.position[3]))
        return
        
    def take_action(self, int_action):
        
        T = 2. * (int(int_action / 3) - 1) #torque applied to handle bar
        d = 0.02 * (int_action % 3 - 1)  #displacement of center of mass

        if self.noise > 0:
            d += (np.random.random() - 0.5) * self.noise
            
        omega, omega_dot, omega_ddot, theta, theta_dot = tuple(self.state) #theta - handle bar, omega - angle of bicycle to verticle
        x_f, y_f, x_b, y_b, psi = tuple(self.position)
        
        for step in range(self.sim_steps):  #Each time we make 10 steps
            
            if theta == 0:
                r_f = r_b = r_cm = 1.e8
            else:
                r_f = self.l / np.abs(np.sin(theta))
                r_b = self.l / np.abs(np.tan(theta))
                r_cm = np.sqrt((self.l - self.c) ** 2 + (self.l ** 2 / np.tan(theta) ** 2))
            
            varphi = omega + np.arctan(d / self.h)
            
            omega_ddot = self.m * self.h * self.gravity * np.sin(varphi)
            omega_ddot -= np.cos(varphi) * (self.inertia_dc * self.sigma_dot * theta_dot + np.sign(theta) * (self.v * self.v * (self.m_d * self.r * (1 / r_f + 1 / r_b) + self.m * self. h / r_cm )))
            omega_ddot /= self.inertia_bc
            
            theta_ddot = (T - self.inertia_dv * self.sigma_dot * omega_dot) / self.inertia_dl
            
            df = (self.delta_time / float(self.sim_steps))
            omega_dot += df * omega_ddot
            omega += df * omega_dot
            theta_dot += df * theta_ddot
            theta += df * theta_dot
            
            #Clipping the maximum possible values of theta
            theta = np.clip(theta, self.theta_range[0], self.theta_range[1])
            
            #Updating position of tires
            front_term = psi + theta + np.sign(psi + theta) * np.arcsin(np.clip(self.v * df / (2 * r_f), -1., 1.))
            back_term = psi + np.sign(psi) * np.arcsin(np.clip(self.v * df / (2.*r_b), -1., 1.))
            x_f += -np.sin(front_term)  
            y_f += np.cos(front_term)
            x_b += -np.sin(back_term)
            y_b += np.cos(back_term)
            
            # Handle Roundoff errors, to keep the length of the bicycle constant
            dist = np.sqrt( (x_f - x_b) ** 2 + (y_f - y_b) ** 2)
            if np.abs(dist - self.l) > 0.01:
                #Need more clarity on why this formula and how it solves the problem
                x_b += (x_b - x_f) * (self.l - dist)/dist
                y_b += (y_b - y_f) * (self.l - dist)/dist
                
            
            # Update psi
            if x_f == x_b and y_f - y_b < 0:
                psi = np.pi
            elif y_f - y_b > 0:
                psi = np.arctan((x_b - x_f) / (y_f - y_b))
            else:
                psi = np.sign(x_b - x_f) * (np.pi / 2.) - np.arctan((y_f - y_b)/(x_b-x_f))
            

        self.state = np.array([omega, omega_dot, omega_ddot, theta, theta_dot])
        self.position = np.array([x_f, y_f, x_b, y_b, psi])
            
        reward, self.end = self.get_reward(), self.is_end()
            
        return (self.state, reward, self.end)
            
    
    def get_reward(self):
        if self.is_end():
            if self.is_at_goal():
                return self.reward_goal
            else:   #Cycle has fallen
                return self.reward_fall
        else:   #Shaping
            x_f, y_f, x_b, y_b = self.position[0], self.position[1], self.position[2], self.position[3]
            goal_angle = vector_angle(self.goal_loc, np.array([x_f-x_b, y_f-y_b]))
            return (4. - goal_angle**2) * self.reward_shaping
    
    def is_at_goal(self):
        dist_btw_goal = np.sqrt(max(0.,((self.position[:2] - self.goal_loc)**2).sum() - self.goal_rsqrd))
        
        if dist_btw_goal < 1.e-5:
            return True
        else:
            return False
        
    def is_end(self):
        omega = self.state[0]
        if (omega < self.omega_range[0]) or (omega > self.omega_range[1]):   #if bicycle falls an angle greater than +/- 12 degrees, it falls
            return True
        
        if self.is_at_goal():  #If the goal is reached
            return True
    
        return False
    


def vector_angle(vec1, vec2):
    if vec1.shape != vec2.shape:
        return (None)
    else:
        cos_angle =  (sum(vec1 * vec2) / (np.sqrt(sum(vec1 * vec1)) * np.sqrt(sum(vec2 * vec2))))
        return (np.arccos(cos_angle))




def make_neural_form(state):
	# Takes a tuple input of states and makes it into the form of 1 X 3456 for input to neural network.
    omega, omega_dot, omega_ddot, theta, theta_dot = tuple(state) #theta - handle bar, omega - angle of bicycle to verticle
    inp = np.zeros(3456, np.float32)
    
    bucket = np.zeros(5)
    
    if theta >= -np.pi/2 and theta < -1:
        bucket[0] = 0
    elif theta >= -1 and theta < -0.2:
        bucket[0] = 1
    elif theta >= -0.2 and theta < 0:
        bucket[0] = 2
    elif theta >= 0 and theta < 0.2:
        bucket[0] = 3
    elif theta >= 0.2 and theta < 1:
        bucket[0] = 4
    elif theta >= 1 and theta <= np.pi/2:
        bucket[0] = 5
    
    if theta_dot < -2:
        bucket[1] = 0
    elif theta_dot >= -2 and theta_dot < 0:
        bucket[1] = 1
    elif theta_dot >= 0 and theta_dot < 2:
        bucket[1] = 2
    elif theta_dot >= 2:
        bucket[1] = 3
        
    if omega >= -np.pi/15 and omega < -0.15:
        bucket[2] = 0
    elif omega >= -0.15 and omega < -0.06:
        bucket[2] = 1
    elif omega >= -0.06 and omega < 0:
        bucket[2] = 2
    elif omega >= 0 and omega < 0.06:
        bucket[2] = 3
    elif omega >= 0.06 and omega < 0.15:
        bucket[2] = 4
    elif omega >= 0.15 and omega <= np.pi/15:
        bucket[2] = 5
        
    if omega_dot < -0.5:
        bucket[3] = 0
    elif omega_dot >= -0.5 and omega_dot < -0.25:
        bucket[3] = 1
    elif omega_dot >= -0.25 and omega_dot < 0:
        bucket[3] = 2
    elif omega_dot >= 0 and omega_dot < 0.25:
        bucket[3] = 3
    elif omega_dot >= 0.25 and omega_dot < 0.5:
        bucket[3] = 4
    elif omega_dot >= 0.5:
        bucket[3] = 5
        
    if omega_ddot < -2:
        bucket[4] = 0
    elif omega_ddot >= -2 and omega_ddot < 0:
        bucket[4] = 1
    elif omega_ddot >= 0 and omega_ddot < 2:
        bucket[4] = 2
    elif omega_ddot >= 2:
        bucket[4] = 3
        
    index = int(bucket[0]*576 + bucket[1]*144 + bucket[2]*24 + bucket[3]*4 + bucket[4])
    inp[index] = 1.0

    
    return torch.from_numpy(inp)



def make_action(state):
	#Gets a state as input and makes action.
    state = make_neural_form(state).to(device)
    output = agent(state)
    if np.random.binomial(1, 0.3, 1):  #e-greedy policy with e = 0.3
        action = np.random.choice(np.arange(0, 9, 1))
    else:
        action = torch.argmax(output)
        
    q_value_state_action = output[action]
    return action, q_value_state_action

if __name__ == "__main__":
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('Using device:', device)

	#Additional Info when using cuda
	if device.type == 'cuda':
		print(torch.cuda.get_device_name(0))
		print('Memory Usage:')
		print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
		print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')



	input_dim, output_dim = 3456, 9
	gamma = 0.9


	agent = torch.nn.Linear( input_dim, output_dim ).cuda()
	loss = torch.nn.MSELoss()
	optimizer = torch.optim.SGD(agent.parameters(), lr=1e-2)

	weights_before = []
	for param in agent.parameters():
		weights_before.append(param.clone())
		
	steps = []
	cycle = Bicycle()

	print_episode = False

	for episode in range(1, 1000+1):
		cycle.reset()

		end = False
		total_reward = 0
		
		while not end:
			state = cycle.state
			action, q_state_action = make_action(state)
			next_state, reward, end = cycle.take_action(action.item())
			total_reward += reward
			
			_, q_next_state_action = make_action(next_state)
			
			td_error = reward + gamma * q_next_state_action - q_state_action

			optimizer.zero_grad()
			td_error.backward(retain_graph=True)
			optimizer.step()    

