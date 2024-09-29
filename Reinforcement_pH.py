
import streamlit as st
import numpy as np
import pandas as pd
from collections import namedtuple
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import random
import io
import inspect
import os

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = Experience(state, action, reward, next_state, done)
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, alpha=0.6, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

        probs = prios ** alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

def create_custom_model(input_dim, output_dim, layers):
    model = Sequential([Dense(layers[0], input_dim=input_dim, activation='relu')])
    for neurons in layers[1:]:
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))
    return model

class DQNAgent:
    def __init__(self, input_dim, output_dim, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, gamma=0.5, action_space=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.action_space = action_space  # Added action space
        self.model = None  # Will be set in main()
        self.target_model = None  # Will be set in main()
        self.memory = PrioritizedReplayBuffer(10000)
        self.last_action_moved_away = False
        self.last_action_type = None
        self.consecutive_negative_rewards = 0
        self.batch_size = 32


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state, current_pH, desired_pH, tolerance, action_space):
        if abs(current_pH - desired_pH) <= tolerance:
            st.write("Within desired pH range, no adjustment needed.")
            self.consecutive_negative_rewards = 0
            return np.zeros(len(action_space[0]))  # Return zero action if within tolerance

        explore_probability = self.epsilon
        if self.consecutive_negative_rewards > 5:
            explore_probability = max(self.epsilon, min(0.5, self.consecutive_negative_rewards * 0.1))

        if np.random.rand() <= explore_probability:
            # Exploration: Random action
            action = random.choice(action_space)
            st.write("Exploring")
        else:
            # Exploitation: Choose the best action based on Q-values
            q_values = self.model.predict(state)[0]
            num_actions = len(action_space)
            
            # Find the index of the action with the maximum Q-value
            action_index = np.argmax(q_values[:num_actions])
            
            # Map the index to the action in the action space
            action = action_space[action_index]
            st.write("Exploiting")

        st.write(f"Action chosen: {action}")
        return np.array(action)



    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])

        target_f = self.model.predict(state)
        action_index = sum((action[i] - action_ranges[i][0]) / (action_ranges[i][1] - action_ranges[i][0]) * (5**i)
                           for i in range(len(action_ranges)))
        action_index = int(action_index)
        target_f[0][action_index] = target

        self.model.fit(state, target_f, epochs=1, verbose=0)

        if reward < 0:
            self.consecutive_negative_rewards += 1
        else:
            self.consecutive_negative_rewards = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = np.array(states).reshape(self.batch_size, -1)
        next_states = np.array(next_states).reshape(self.batch_size, -1)

        # Get current and next Q-values
        current_q_values = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)

        num_actions = len(self.action_space)

        for i in range(self.batch_size):
            # Find the index of the action in the action space
            action_array = np.array(self.action_space)
            action_index = np.where(np.all(action_array == actions[i], axis=1))[0][0]

            # Compute the target Q-value
            if dones[i]:
                target = rewards[i]
            else:
                target = rewards[i] + self.gamma * np.max(next_q_values[i])

            # Update the Q-value in the current Q-values matrix
            current_q_values[i][action_index] = target

        # Fit the model
        self.model.fit(states, current_q_values, epochs=1, verbose=0, sample_weight=weights)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay




def predict_pH(current_state_array, ph_model, poly_gbr_model, state_scaler, poly_gbr_scaler, poly):
    current_state_array = np.reshape(current_state_array, (1, -1))
    print(current_state_array)
    scaled_features = state_scaler.transform(current_state_array)
    predicted_online_pH = ph_model.predict(scaled_features)[0][0]
    print(f"Predicted Online pH: {predicted_online_pH}")
    current_state_array_with_online_Ph = np.append(current_state_array, predicted_online_pH).reshape(1, -1)
    scaled_current_state = poly.fit_transform(current_state_array_with_online_Ph)
    scaled_current_state = poly_gbr_scaler.transform(scaled_current_state)
    predicted_offline_pH = poly_gbr_model.predict(scaled_current_state)[0]
    print(f"Predicted Offline pH: {predicted_offline_pH}")
    return predicted_offline_pH

def predict_pH_online(current_state_array, ph_model, state_scaler):
    current_state_array = np.reshape(current_state_array, (1, -1))
    scaled_features = state_scaler.transform(current_state_array)
    predicted_online_pH = ph_model.predict(scaled_features)[0][0]
    print(f"Predicted Online pH: {predicted_online_pH}")
    st.write(f"Predicted Online pH: {predicted_online_pH}")
    return predicted_online_pH

def modified_control_loop(agent, initial_state, data_df, desired_pH, control_keys, target_key, ph_model, poly_gbr_model, state_scaler, poly_gbr_scaler, poly, action_ranges, calculate_reward):
    time_step_interval = 4
    timesteps = 0
    batch_size = 10
    current_state = initial_state.copy()
    tolerance = 0.1

    # Initialize volumes for control keys
    control_volumes = {key: 1.0 for key in control_keys}

    # Results dictionary
    results = {
        'time_step_list': [],
        'predicted_pH_list': [],
        'reward_list': [],
        'actual_pH_list': [],
        'predicted_online_pH': [],
    }
    for key in control_keys:
        results[f'{key}_vol'] = []

    total_steps = len(data_df)

    # Create a placeholder for live updates
    plot_placeholder = st.empty()

    for index, row in data_df.iterrows():
        current_state = row.to_dict()
        results['actual_pH_list'].append(current_state[target_key])

        # Set the control volumes based on the current state
        for key in control_keys:
            current_state[key] = control_volumes[key]

        current_time = current_state['Rel. Time (in s)']

        state_without_target = []

        for key in current_state:
            if key != target_key:
                state_without_target.append(current_state[key])

        state_array = np.array(state_without_target)

        # Predict pH values before action using ph_model and poly_gbr_model
        predicted_pH_before_action = predict_pH(state_array, ph_model, poly_gbr_model, state_scaler, poly_gbr_scaler, poly)
        predict_online = predict_pH_online(state_array, ph_model, state_scaler)

        # Append the predicted online pH to the state array
        current_state_array = np.append(state_array, predict_online)
        current_state_array_reshaped = np.reshape(current_state_array, (1, -1))

        # Choose action
        action = agent.choose_action(current_state_array_reshaped, predict_online, desired_pH, tolerance, action_ranges)

        if np.all(action == 0):
            results['predicted_online_pH'].append(predict_online)
            results['predicted_pH_list'].append(predicted_pH_before_action)
            results['reward_list'].append(0)
            results['time_step_list'].append(current_time)
            for key in control_keys:
                results[f'{key}_vol'].append(control_volumes[key])
        else:
            # Update control volumes based on action
            for i, key in enumerate(control_keys):
                control_volumes[key] += action[i]

            # Set updated volumes in the current state
            for key in control_keys:
                current_state[key] = control_volumes[key]

            updated_without_target = []

            for key in current_state:
                if key != target_key:
                    updated_without_target.append(current_state[key])

            updated_array = np.array(updated_without_target)

            # Predict pH values after action using ph_model and poly_gbr_model
            predicted_pH_after_action = predict_pH(updated_array, ph_model, poly_gbr_model, state_scaler, poly_gbr_scaler, poly)
            online = predict_pH_online(updated_array, ph_model, state_scaler)

            updated_state_array = np.append(updated_array, online)

            # Calculate reward
            reward = calculate_reward(online, desired_pH, predict_online, predicted_pH_after_action, predicted_pH_before_action)

            agent.last_action_moved_away = reward < 0
            done = abs(online - desired_pH) <= tolerance

            # Store experience and replay
            agent.remember(current_state_array_reshaped, action, reward, np.reshape(updated_state_array, (1, -1)), done)
            agent.replay()

            if timesteps % 200 == 0:
                agent.update_target_model()

            current_state[target_key] = predicted_pH_after_action

            # Store results
            results['predicted_online_pH'].append(online)
            results['predicted_pH_list'].append(predicted_pH_after_action)
            results['reward_list'].append(reward)
            results['time_step_list'].append(current_time)
            for key in control_keys:
                results[f'{key}_vol'].append(control_volumes[key])

        timesteps += 1

        # Update the plot
        with plot_placeholder.container():
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

            ax1.plot(results['time_step_list'], results['predicted_pH_list'], marker='o', linestyle='-', color='b', label='Predicted pH')
            ax1.plot(results['time_step_list'], results['actual_pH_list'], marker='x', linestyle='--', color='g', label='Actual pH')
            ax1.plot(results['time_step_list'], results['predicted_online_pH'], marker='^', linestyle=':', color='r', label='Predicted Online pH')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('pH')
            ax1.legend(loc='upper left')
            ax1.set_title('pH over Time')

            for key in control_keys:
                ax2.plot(results['time_step_list'], results[f'{key}_vol'], marker='o', linestyle='-', label=f'{key} Volume')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Volume')
            ax2.legend()
            ax2.set_title('Reagent Volumes over Time')

            plt.tight_layout()
            st.pyplot(fig)

    return results


def main():
    st.title("pH Control Simulation with Customizable RL")

    # Model selection
    st.header("Model and Scaler Upload")
    col1, col2 = st.columns(2)
    with col1:
        ph_model_file = st.file_uploader("Upload pH prediction model (PKL file)", type="pkl")
        gbr_model_file = st.file_uploader("Upload GBR model (PKL file)", type="pkl")
    with col2:
        scaler_file = st.file_uploader("Upload state scaler (PKL file)", type="pkl")
        poly_scaler_file = st.file_uploader("Upload poly scaler (PKL file)", type="pkl")

    if ph_model_file and gbr_model_file and scaler_file and poly_scaler_file:
        # Convert UploadedFile to a byte stream
        ph_model_bytes = io.BytesIO(ph_model_file.read())
        ph_model = pickle.load(ph_model_bytes)

        gbr_model_bytes = io.BytesIO(gbr_model_file.read())
        poly_gbr_model = pickle.load(gbr_model_bytes)

        state_scaler_bytes = io.BytesIO(scaler_file.read())
        state_scaler = pickle.load(state_scaler_bytes)

        poly_gbr_scaler_bytes = io.BytesIO(poly_scaler_file.read())
        poly_gbr_scaler = pickle.load(poly_gbr_scaler_bytes)

        poly = PolynomialFeatures(degree=2, include_bias=False)
        st.success("Models and scalers loaded successfully!")
    else:
        st.warning("Please upload all models and scalers to continue.")
        return

    # Data upload
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    
    if uploaded_file is not None:
        data_df = pd.read_excel(uploaded_file)
        st.success("File successfully uploaded and read!")

        st.subheader("Data Preview")
        st.dataframe(data_df.head())

        # Column selection
        st.header("Column Selection")
        all_columns = data_df.columns.tolist()
        
        # Action Space Customization
        st.header("Action Space Customization")
        num_control_vars = st.number_input("Number of control variables", min_value=1, value=2)
        control_keys = st.multiselect(
            f"Select {num_control_vars} control keys",
            all_columns,
            default=["Fällungslauge.TotalVolume", "Metallsulfatlösung.TotalVolume"][:num_control_vars]
        )
        action_ranges = {}
        for key in control_keys:
            action_1 = st.number_input(f"action1{key}", value=0.2)
            action_2 = st.number_input(f"action2{key}", value=0.4)
            action_3 = st.number_input(f"action3{key}", value=0.6)
            action_4 = st.number_input(f"action4{key}", value=0.8)
            action_5 = st.number_input(f"action5{key}", value=1.0)
            
            action_ranges[key]=[action_1,action_2,action_3,action_4,action_5]
        
        lists = [actions for actions in action_ranges.values()]
        grids = np.meshgrid(*lists, indexing='ij')
        action_space = np.stack(grids, axis=-1).reshape(-1, len(action_ranges))

        
        target_key = st.selectbox(
            "Select target key (e.g., pH)",
            all_columns,
            index=all_columns.index("pH") if "pH" in all_columns else 0
        )

        # Reward Function Customization
        st.header("Reward Function Customization")

        st.subheader("Direction Rewards")
        correct_direction_reward = st.slider("Reward for moving in the correct direction", -100.0, 100.0, 50.0)
        wrong_direction_penalty = st.slider("Penalty for moving in the wrong direction", -100.0, 0.0, -50.0)

        st.subheader("Magnitude Rewards")
        magnitude_multiplier = st.slider("Reward multiplier for change magnitude", 0.0, 1000.0, 500.0)

        st.subheader("Target Proximity Rewards")
        proximity_multiplier = st.slider("Reward multiplier for proximity to target", 0.0, 100.0, 50.0)
        target_reached_bonus = st.slider("Bonus for reaching target pH (within tolerance)", 0.0, 200.0, 100.0)

        st.subheader("Other Parameters")
        pH_tolerance = st.slider("pH tolerance for target", 0.01, 0.5, 0.1)

        # Create a reward function using these parameters
        def calculate_reward(online_after, desired_pH, online_before, offline_after, offline_before):
            online_delta = online_after - online_before
            
            # Determine if moving in the right direction
            right_direction = (
                (online_delta > 0 and online_before < desired_pH) or
                (online_delta < 0 and online_before > desired_pH) or
                (abs(online_delta) < 1e-6 and abs(online_before - desired_pH) < pH_tolerance)
            )
            
            # Base reward
            if right_direction:
                reward = correct_direction_reward + abs(online_delta) * magnitude_multiplier
            else:
                reward = wrong_direction_penalty - abs(online_delta) * magnitude_multiplier
            
            # Proximity reward
            distance_reward = -abs(online_after - desired_pH) * proximity_multiplier
            
            # Target reached bonus
            bonus = target_reached_bonus if abs(online_after - desired_pH) < pH_tolerance else 0
            
            total_reward = reward + distance_reward + bonus
            
            return total_reward

        # Display the reward function
        st.subheader("Resulting Reward Function")
        st.code(inspect.getsource(calculate_reward), language="python")

        # Neural Network Customization
        st.header("Neural Network Customization")
        num_layers = st.number_input("Number of hidden layers", min_value=1, value=2)
        layers = []
        for i in range(num_layers):
            neurons = st.number_input(f"Neurons in layer {i+1}", min_value=1, value=64)
            layers.append(neurons)

        # Hyperparameter Customization
        st.header("Hyperparameter Customization")
        epsilon = st.slider("Epsilon", min_value=0.0, max_value=1.0, value=1.0)
        epsilon_decay = st.slider("Epsilon decay", min_value=0.9, max_value=0.9999, value=0.995)
        learning_rate = st.slider("Learning rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
        gamma = st.slider("Gamma (discount factor)", min_value=0.1, max_value=0.99, value=0.5)

        # Desired pH
        desired_pH = st.slider("Select desired pH level", 0.0, 14.0, 7.0, 0.1)

        # Data range selection
        st.subheader("Select Data Range")
        start_index = st.number_input("Start index", min_value=0, max_value=len(data_df)-1, value=0)
        end_index = st.number_input("End index", min_value=start_index+1, max_value=len(data_df), value=min(start_index+1000, len(data_df)))

        # Run simulation button
        if st.button("Run Simulation"):
        # Create custom DQNAgent with action_space
            agent = DQNAgent(
                input_dim=len(all_columns),
                output_dim=5**len(control_keys),  # Adjusted to handle action space
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                gamma=gamma,
                action_space=action_space  # Pass action_space here
            )
            agent.model = create_custom_model(agent.input_dim, agent.output_dim, layers)
            agent.target_model = create_custom_model(agent.input_dim, agent.output_dim, layers)
            agent.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
            agent.target_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

            # Prepare initial state
            initial_state = data_df.iloc[start_index][control_keys + [target_key]].to_dict()

            # Run the modified control loop
            results = modified_control_loop(
                agent, initial_state, data_df.iloc[start_index:end_index], desired_pH,
                control_keys, target_key, ph_model, poly_gbr_model, state_scaler, poly_gbr_scaler, poly,
                action_space, calculate_reward
            )

            # Plot results
            st.subheader("Simulation Results")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

            ax1.plot(results['time_step_list'], results['predicted_pH_list'], marker='o', linestyle='-', color='b', label='Predicted pH')
            ax1.plot(results['time_step_list'], results['actual_pH_list'], marker='x', linestyle='--', color='g', label='Actual pH')
            ax1.plot(results['time_step_list'], results['predicted_online_pH'], marker='^', linestyle=':', color='r', label='Predicted Online pH')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('pH')
            ax1.legend(loc='upper left')
            ax1.set_title('pH over Time')

            for key in control_keys:
                ax2.plot(results['time_step_list'], results[f'{key}_vol'], marker='o', linestyle='-', label=f'{key} Volume')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Volume')
            ax2.legend()
            ax2.set_title('Reagent Volumes over Time')

            plt.tight_layout()
            st.pyplot(fig)

            # Display metrics
            st.subheader("Performance Metrics")
            average_reward = np.mean(results['reward_list'])
            st.metric("Average Reward", f"{average_reward:.2f}")

            final_pH_difference = abs(results['predicted_pH_list'][-1] - desired_pH)
            st.metric("Final pH Difference", f"{final_pH_difference:.2f}")

            # Download results
            results_df = pd.DataFrame(results)
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="simulation_results.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()

