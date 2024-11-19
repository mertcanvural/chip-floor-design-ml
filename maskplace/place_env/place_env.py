import math
import gym
from gym import spaces
import numpy as np
import sys
sys.path.append("..")
from place_db import PlaceDB
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

class PlaceEnv(gym.Env):
    def __init__(self, placedb, placed_num_macro=None, grid=224):
        # Initialization
        print("grid * grid", grid * grid)
        print("placedb.node_cnt", placedb.node_cnt)
        print("placedb.net_cnt", placedb.net_cnt)
        assert grid * grid >= placedb.node_cnt  # Ensure the grid is large enough

        # Environment attributes
        self.grid = grid
        self.max_height = placedb.max_height
        self.max_width = placedb.max_width
        self.placedb = placedb
        self.num_macro = placedb.node_cnt
        self.placed_num_macro = placed_num_macro
        self.num_net = placedb.net_cnt
        self.node_name_list = placedb.node_id_to_name
        self.action_space = spaces.Discrete(self.grid * self.grid)

        # Initialize placeholders
        self.state = None  # This will be initialized in reset()
        self.net_min_max_ord = {}
        self.node_pos = {}
        self.net_placed_set = {}
        self.last_reward = 0
        self.num_macro_placed = 0
        self.node_x_max = 0
        self.node_x_min = self.grid
        self.node_y_max = 0
        self.node_y_min = self.grid
        self.ratio = self.placedb.max_height / self.grid
        print("self.ratio = {:.2f}".format(self.ratio))

        # Define a placeholder observation space with a dummy shape
        # Update this in reset() when the state is initialized
        # dummy_state_length = 1 + grid * grid * 5 + 2  # Adjust if necessary
        # Define observation_space
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(grid * grid * 6 + 3,),  # Adjust based on your `self.state` shape
            dtype=np.float32
        )


    
    def reset(self):
        # Reset macro placement count
        self.num_macro_placed = 0
        num_macro = self.num_macro
        
        # Initialize canvas and positional data structures
        canvas = np.zeros((self.grid, self.grid))
        self.node_pos = {}
        self.net_min_max_ord = {}
        self.net_fea = np.zeros((self.num_net, 4))
        self.net_fea[:, 0] = 0
        self.net_fea[:, 1] = 1.0
        self.net_fea[:, 2] = 0
        self.net_fea[:, 3] = 1.0
        self.rudy = np.zeros((self.grid, self.grid))
        
        # Process ports and nets
        for port_name in self.placedb.port_to_net_dict:
            for net_name in self.placedb.port_to_net_dict[port_name]:
                pin_x = round(self.placedb.port_info[port_name]['x'] / self.ratio)
                pin_y = round(self.placedb.port_info[port_name]['y'] / self.ratio)
                
                if net_name in self.net_min_max_ord:
                    if pin_x > self.net_min_max_ord[net_name]['max_x']:
                        self.net_min_max_ord[net_name]['max_x'] = pin_x
                        self.net_fea[self.placedb.net_info[net_name]['id']][1] = pin_x / self.grid
                    elif pin_x < self.net_min_max_ord[net_name]['min_x']:
                        self.net_min_max_ord[net_name]['min_x'] = pin_x
                        self.net_fea[self.placedb.net_info[net_name]['id']][0] = pin_x / self.grid
                    
                    if pin_y > self.net_min_max_ord[net_name]['max_y']:
                        self.net_min_max_ord[net_name]['max_y'] = pin_y
                        self.net_fea[self.placedb.net_info[net_name]['id']][3] = pin_y / self.grid
                    elif pin_y < self.net_min_max_ord[net_name]['min_y']:
                        self.net_min_max_ord[net_name]['min_y'] = pin_y
                        self.net_fea[self.placedb.net_info[net_name]['id']][2] = pin_y / self.grid
                else:
                    self.net_min_max_ord[net_name] = {
                        'max_x': pin_x,
                        'min_x': pin_x,
                        'max_y': pin_y,
                        'min_y': pin_y
                    }
                    self.net_fea[self.placedb.net_info[net_name]['id']][1] = pin_x / self.grid
                    self.net_fea[self.placedb.net_info[net_name]['id']][0] = pin_x / self.grid
                    self.net_fea[self.placedb.net_info[net_name]['id']][3] = pin_y / self.grid
                    self.net_fea[self.placedb.net_info[net_name]['id']][2] = pin_y / self.grid
        
        self.net_placed_set = {}
        self.num_macro_placed = 0
        
        # Initialize images and masks
        net_img = np.zeros((self.grid, self.grid))
        net_img_2 = np.zeros((self.grid, self.grid))

        # Calculate dimensions for the current and next macro
        next_x = math.ceil(max(1, self.placedb.node_info[self.node_name_list[self.num_macro_placed]]['x'] / self.ratio))
        next_y = math.ceil(max(1, self.placedb.node_info[self.node_name_list[self.num_macro_placed]]['y'] / self.ratio))
        mask = self.get_mask(canvas, next_x, next_y)

        # Ensure `self.num_macro_placed + 1` does not exceed the total number of macros
        if self.num_macro_placed + 1 < len(self.node_name_list):
            next_x_2 = math.ceil(max(1, self.placedb.node_info[self.node_name_list[self.num_macro_placed + 1]]['x'] / self.ratio))
            next_y_2 = math.ceil(max(1, self.placedb.node_info[self.node_name_list[self.num_macro_placed + 1]]['y'] / self.ratio))
            mask_2 = self.get_mask(canvas, next_x_2, next_y_2)
        else:
            next_x_2 = 0
            next_y_2 = 0
            mask_2 = np.zeros((self.grid, self.grid))
        
        # Initialize net placement sets
        for net_name in self.placedb.net_info:
            self.net_placed_set[net_name] = set()
        
        # Combine state features
        self.state = np.concatenate((
            np.array([self.num_macro_placed]),
            canvas.flatten(),
            net_img.flatten(),
            mask.flatten(),
            net_img_2.flatten(),
            mask_2.flatten(),
            np.array([next_x / self.grid, next_y / self.grid])
        ), axis=0)

        # Reset placement bounds
        self.node_x_max = 0
        self.node_x_min = self.grid
        self.node_y_max = 0
        self.node_y_min = self.grid
        
        # Define observation space based on state
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state.shape[0],),
            dtype=np.float32
        )
        
        return self.state


    def save_fig(self, file_path):
        # Create a new figure and axis
        fig1, ax1 = plt.subplots(figsize=(10, 10))
        ax1.set_aspect('equal', adjustable='box')
        ax1.axis('off')  # Turn off the axes

        # Add rectangles for each node in the placement
        for node_name in self.node_pos:
            x, y, size_x, size_y = self.node_pos[node_name]
            rect = patches.Rectangle(
                (x / self.grid, y / self.grid),  # Bottom-left corner
                size_x / self.grid,             # Width
                size_y / self.grid,             # Height
                linewidth=1,
                edgecolor='black',
                facecolor='none'                # Transparent fill
            )
            ax1.add_patch(rect)
        
        # Save the figure to the specified file path
        fig1.savefig(file_path, dpi=90, bbox_inches='tight')
        plt.close(fig1)  # Close the figure to free resources

    # WireMask
    def get_net_img(self, is_next_next=False):
        # Initialize the net image as a grid of zeros
        net_img = np.zeros((self.grid, self.grid))
        
        # Determine the next node to process
        if not is_next_next:
            next_node_name = self.placedb.node_id_to_name[self.num_macro_placed]
        elif self.num_macro_placed + 1 < len(self.placedb.node_id_to_name):
            next_node_name = self.placedb.node_id_to_name[self.num_macro_placed + 1]
        else:
            return net_img  # Return empty grid if no next node exists

        # Process each net connected to the current node
        for net_name in self.placedb.node_to_net_dict[next_node_name]:
            if net_name in self.net_min_max_ord:
                # Calculate pin offsets and net boundary changes
                delta_pin_x = round(
                    (self.placedb.node_info[next_node_name]['x'] / 2 +
                    self.placedb.net_info[net_name]["nodes"][next_node_name]["x_offset"]) / self.ratio
                )
                delta_pin_y = round(
                    (self.placedb.node_info[next_node_name]['y'] / 2 +
                    self.placedb.net_info[net_name]["nodes"][next_node_name]["y_offset"]) / self.ratio
                )
                start_x = max(0, self.net_min_max_ord[net_name]['min_x'] - delta_pin_x)
                end_x = min(self.grid - 1, self.net_min_max_ord[net_name]['max_x'] - delta_pin_x)
                start_y = max(0, self.net_min_max_ord[net_name]['min_y'] - delta_pin_y)
                end_y = min(self.grid - 1, self.net_min_max_ord[net_name]['max_y'] - delta_pin_y)

                # Default weight is 1.0 unless specified
                weight = self.placedb.net_info[net_name].get('weight', 1.0)

                # Update net image based on net boundaries and weights
                for i in range(0, start_x):
                    net_img[i, :] += (start_x - i) * weight
                for i in range(end_x + 1, self.grid):
                    net_img[i, :] += (i - end_x) * weight
                for j in range(0, start_y):
                    net_img[:, j] += (start_y - j) * weight
                for j in range(end_y + 1, self.grid):
                    net_img[:, j] += (j - start_y) * weight

        # Return the generated net image
        return net_img

    def step(self, action):
        # Validate action
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        # Parse current canvas and mask
        canvas = self.state[1: 1 + self.grid * self.grid].reshape(self.grid, self.grid)
        mask = self.state[1 + self.grid * self.grid * 2: 1 + self.grid * self.grid * 3].reshape(self.grid, self.grid)
        reward = 0

        # Translate action into grid coordinates
        x = round(action // self.grid)
        y = round(action % self.grid)

        # Penalize invalid action (if it's masked)
        if mask[x][y] == 1:
            reward += -200000

        # Fetch the node to be placed
        node_name = self.placedb.node_id_to_name[self.num_macro_placed]
        size_x = math.ceil(max(1, self.placedb.node_info[node_name]['x'] / self.ratio))
        size_y = math.ceil(max(1, self.placedb.node_info[node_name]['y'] / self.ratio))

        # Ensure size consistency
        assert abs(size_x - self.state[-2] * self.grid) < 1e-5
        assert abs(size_y - self.state[-1] * self.grid) < 1e-5

        # Update canvas with the new placement
        canvas[x:x + size_x, y:y + size_y] = 1.0
        canvas[x:x + size_x, y] = 0.5
        if y + size_y - 1 < self.grid:
            canvas[x:x + size_x, max(0, y + size_y - 1)] = 0.5
        canvas[x, y:y + size_y] = 0.5
        if x + size_x - 1 < self.grid:
            canvas[max(0, x + size_x - 1), y:y + size_y] = 0.5

        # Track placement
        self.node_pos[self.node_name_list[self.num_macro_placed]] = (x, y, size_x, size_y)

        # Process nets connected to the current node
        for net_name in self.placedb.node_to_net_dict[node_name]:
            self.net_placed_set[net_name].add(node_name)
            pin_x = round((x * self.ratio + self.placedb.node_info[node_name]['x'] / 2 +
                          self.placedb.net_info[net_name]["nodes"][node_name]["x_offset"]) / self.ratio)
            pin_y = round((y * self.ratio + self.placedb.node_info[node_name]['y'] / 2 +
                          self.placedb.net_info[net_name]["nodes"][node_name]["y_offset"]) / self.ratio)
            if net_name in self.net_min_max_ord:
                start_x = self.net_min_max_ord[net_name]['min_x']
                end_x = self.net_min_max_ord[net_name]['max_x']
                start_y = self.net_min_max_ord[net_name]['min_y']
                end_y = self.net_min_max_ord[net_name]['max_y']
                delta_x = end_x - start_x
                delta_y = end_y - start_y
                if delta_x > 0 or delta_y > 0:
                    self.rudy[start_x:end_x + 1, start_y:end_y + 1] -= 1 / (delta_x + 1) + 1 / (delta_y + 1)
                weight = self.placedb.net_info[net_name].get('weight', 1.0)

                # Update min/max boundaries and calculate rewards
                if pin_x > self.net_min_max_ord[net_name]['max_x']:
                    reward += weight * (self.net_min_max_ord[net_name]['max_x'] - pin_x)
                    self.net_min_max_ord[net_name]['max_x'] = pin_x
                    self.net_fea[self.placedb.net_info[net_name]['id']][1] = pin_x / self.grid
                elif pin_x < self.net_min_max_ord[net_name]['min_x']:
                    reward += weight * (pin_x - self.net_min_max_ord[net_name]['min_x'])
                    self.net_min_max_ord[net_name]['min_x'] = pin_x
                    self.net_fea[self.placedb.net_info[net_name]['id']][0] = pin_x / self.grid
                if pin_y > self.net_min_max_ord[net_name]['max_y']:
                    reward += weight * (self.net_min_max_ord[net_name]['max_y'] - pin_y)
                    self.net_min_max_ord[net_name]['max_y'] = pin_y
                    self.net_fea[self.placedb.net_info[net_name]['id']][3] = pin_y / self.grid
                elif pin_y < self.net_min_max_ord[net_name]['min_y']:
                    reward += weight * (pin_y - self.net_min_max_ord[net_name]['min_y'])
                    self.net_min_max_ord[net_name]['min_y'] = pin_y
                    self.net_fea[self.placedb.net_info[net_name]['id']][2] = pin_y / self.grid
                start_x = self.net_min_max_ord[net_name]['min_x']
                end_x = self.net_min_max_ord[net_name]['max_x']
                start_y = self.net_min_max_ord[net_name]['min_y']
                end_y = self.net_min_max_ord[net_name]['max_y']
                delta_x = end_x - start_x
                delta_y = end_y - start_y
                self.rudy[start_x:end_x + 1, start_y:end_y + 1] += 1 / (delta_x + 1) + 1 / (delta_y + 1)
            else:
                # Initialize net boundaries
                self.net_min_max_ord[net_name] = {
                    'max_x': pin_x, 'min_x': pin_x, 'max_y': pin_y, 'min_y': pin_y
                }
                self.net_fea[self.placedb.net_info[net_name]['id']][1] = pin_x / self.grid
                self.net_fea[self.placedb.net_info[net_name]['id']][0] = pin_x / self.grid
                self.net_fea[self.placedb.net_info[net_name]['id']][3] = pin_y / self.grid
                self.net_fea[self.placedb.net_info[net_name]['id']][2] = pin_y / self.grid

        self.num_macro_placed += 1
        net_img = np.zeros((self.grid, self.grid))
        net_img_2 = np.zeros((self.grid, self.grid))

        # Update images for current and next nodes
        if self.num_macro_placed < self.placed_num_macro:
            net_img = self.get_net_img()
            net_img_2 = self.get_net_img(is_next_next=True)
            if net_img.max() > 0 or net_img_2.max() > 0:
                net_img /= max(net_img.max(), net_img_2.max())
                net_img_2 /= max(net_img.max(), net_img_2.max())

        # Update node boundaries
        self.node_x_max = max(self.node_x_max, x)
        self.node_x_min = min(self.node_x_min, x)
        self.node_y_max = max(self.node_y_max, y)
        self.node_y_min = min(self.node_y_min, y)

        # Determine if placement is complete
        done = self.num_macro_placed == self.num_macro or (
            self.placed_num_macro is not None and self.num_macro_placed == self.placed_num_macro
        )

        # Update masks for the next macro placement
        mask = np.ones((self.grid, self.grid))
        mask_2 = np.ones((self.grid, self.grid))
        if not done:
            next_x = math.ceil(max(1, self.placedb.node_info[self.placedb.node_id_to_name[self.num_macro_placed]]['x'] / self.ratio))
            next_y = math.ceil(max(1, self.placedb.node_info[self.placedb.node_id_to_name[self.num_macro_placed]]['y'] / self.ratio))
            mask = self.get_mask(canvas, next_x, next_y)
            if self.num_macro_placed + 1 < self.placed_num_macro:
                next_x_2 = math.ceil(max(1, self.placedb.node_info[self.placedb.node_id_to_name[self.num_macro_placed + 1]]['x'] / self.ratio))
                next_y_2 = math.ceil(max(1, self.placedb.node_info[self.placedb.node_id_to_name[self.num_macro_placed + 1]]['y'] / self.ratio))
                mask_2 = self.get_mask(canvas, next_x_2, next_y_2)
        else:
            next_x = 0
            next_y = 0

        # Update the state
        self.state = np.concatenate((
            np.array([self.num_macro_placed]),
            canvas.flatten(),
            net_img.flatten(),
            mask.flatten(),
            net_img_2.flatten(),
            mask_2.flatten(),
            np.array([next_x / self.grid, next_y / self.grid])
        ), axis=0)

        # Return the updated state, reward, completion status, and auxiliary info
        return self.state, reward, done, {"raw_reward": reward, "net_img": net_img, "mask": mask}


    # PositionMask
    def get_mask(self, canvas, next_x, next_y):
        """
        Generates a placement mask for the next macro based on the current canvas.

        Args:
            canvas (ndarray): The current placement canvas.
            next_x (int): The width of the next macro to be placed.
            next_y (int): The height of the next macro to be placed.

        Returns:
            mask (ndarray): A 2D grid where invalid placement areas are marked as 1 and valid areas as 0.
        """
        mask = np.zeros((self.grid, self.grid))

        # Mask invalid areas based on already placed nodes
        for node_name in self.node_pos:
            startx = max(0, self.node_pos[node_name][0] - next_x + 1)
            starty = max(0, self.node_pos[node_name][1] - next_y + 1)
            endx = min(self.node_pos[node_name][0] + self.node_pos[node_name][2] - 1, self.grid - 1)
            endy = min(self.node_pos[node_name][1] + self.node_pos[node_name][3] - 1, self.grid - 1)
            mask[startx:endx + 1, starty:endy + 1] = 1

        # Mask areas that exceed the canvas boundary
        mask[self.grid - next_x + 1:, :] = 1
        mask[:, self.grid - next_y + 1:] = 1

        return mask


    