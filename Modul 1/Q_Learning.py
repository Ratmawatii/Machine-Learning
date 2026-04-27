import numpy as np

#Definisi environment (maze 3x3)
# 0 = jalan, 1 = tembok, 2 = goal
maze = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 2]
])

#iniliasisasi Q-table (states x actions)
q_table = np.zeros((9,4))  #9 states 3x3, 4 actions (up, down, left, right)

#Hyperparameters
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.2  # exploration rate

#Training loop
for episode in range(1000):
    state = 0  # start at (0,0)
    done = False
    
    while not done:
        #epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, 4)  # eksplorasi
        else:
            action = np.argmax(q_table[state])  # eksploitasi
        
        #Excecute action
        next_state, reward, done = take_action(state, action, maze)
        
        #Update Q-table
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state