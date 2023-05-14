import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.Visualizer.StateViz import StateViz
from pyRDDLGym import Visualizer


import matplotlib.pyplot as plt


class ArmVisualizer(StateViz):

    def __init__(self, model: PlanningModel,
                 dpi=30,
                 fontsize=8,
                 display=False) -> None:
        self._model = model
        self._states = model.groundstates()
        self._nonfluents = model.groundnonfluents()
        self._objects = model.objects
        self._figure_size = None
        self._dpi = dpi
        self._fontsize = fontsize
        self._interval = 5

        self._asset_path = "/".join(Visualizer.__file__.split("/")[:-1])

        self._nonfluent_layout = None
        self._state_layout = None
        self._shelf_number = 0
        self._can_number = 0
        self._fig, self._ax = None, None
        self._data = None
        self._img = None

    def build_nonfluents_layout(self):
        can_sizes = [[0,0] for _ in range(self._can_number)]
        shelf_sizes = [[0,0,0,0] for _ in range(self._shelf_number)]

        for k, v in self._nonfluents.items():
            var, objects = self._model.parse(k)
            if var=="SIZE_X_c":
                can_sizes[int(objects[0][-1])][0] = v
            elif var == "SIZE_Y_c":
                can_sizes[int(objects[0][-1])][1] = v
            elif var == "MIN_X":
                shelf_sizes[int(objects[0][-1])-1][0] = v
            elif var == "MAX_X":
                shelf_sizes[int(objects[0][-1])-1][1] = v
            elif var == "MIN_Y":
                shelf_sizes[int(objects[0][-1])-1][2] = v
            elif var == "MAX_Y":
                shelf_sizes[int(objects[0][-1])-1][3] = v
        
        return {"can_sizes": can_sizes, "shelf_sizes": shelf_sizes}
    
    def build_states_layout(self, state):
        arm_coordinates = [0,0]
        can_coordinates = [[0,0] for _ in range(self._can_number)]
        can_on_shelf = [[] for _ in range(self._shelf_number)]
        working_shelf = None
        holding = None

        for k, v in state.items():
            var, objects = self._model.parse(k)
            if var == "x_position_a":
                arm_coordinates[0] = v
            elif var == "y_position_a":
                arm_coordinates[1] = v
            elif var == "working-shelf":
                if v:
                    working_shelf = objects[0]
            elif var == "x_position_c":
                can_coordinates[int(objects[0][-1])][0] = v
            elif var == "y_position_c":
                can_coordinates[int(objects[0][-1])][1] = v
            elif var == "on-shelf":
                if v:
                    can_on_shelf[int(objects[-1][-1])-1].append(objects[0])
            elif var == "holding":
                if v:
                    holding = objects[0]

        return {"arm_coordinates": arm_coordinates, "can_coordinates": can_coordinates, "can_on_shelf": can_on_shelf, "working_shelf": working_shelf, "holding": holding}


    def plot_state(self):
        # Extract information
        arm_coordinates = self._state_layout['arm_coordinates']
        can_coordinates = self._state_layout['can_coordinates']
        can_on_shelf = self._state_layout['can_on_shelf']
        working_shelf = self._state_layout['working_shelf']
        holding = self._state_layout['holding']
        can_sizes = self._nonfluent_layout['can_sizes']
        shelf_sizes = self._nonfluent_layout['shelf_sizes']

        print(can_on_shelf)

        # Clear previous plot
        for ax in self._ax:
            ax.cla()

        # Loop over shelves
        for i in range(self._shelf_number):
            shelf_num = i + 1

            # Draw shelf
            self._ax[i].add_patch(plt.Rectangle((shelf_sizes[i][0], shelf_sizes[i][2]), shelf_sizes[i][1], shelf_sizes[i][3], color='#997950'))

            # Plot arm
            arm_x, arm_y = arm_coordinates
            if working_shelf == f's{shelf_num}':
                self._ax[i].add_patch(plt.Rectangle((arm_x, -6), 1, arm_y + 6, color='black'))

            # Get coordinates of cans on shelf
            cans_on_shelf = can_on_shelf[i]
            for can_name in cans_on_shelf:
                can_index = int(can_name[-1])
                can_x, can_y = can_coordinates[can_index]
                colour = '#F40049'
                self._ax[i].add_patch(plt.Rectangle((can_x, can_y), can_sizes[can_index][0], can_sizes[can_index][1], color=colour))

            if holding and int(working_shelf[-1]) == shelf_num:
                colour = '#DA0016'
                can_index = int(holding[-1])
                can_x, can_y = can_coordinates[can_index]
                self._ax[i].add_patch(plt.Rectangle((can_x, can_y), can_sizes[can_index][0], can_sizes[can_index][1], color=colour))


            # Set axis limits
            self._ax[i].set_xlim([shelf_sizes[i][0], shelf_sizes[i][1]])
            self._ax[i].set_ylim([shelf_sizes[i][2] - 6, shelf_sizes[i][3]])


    def init_canvas(self, figure_size, dpi):
        fig, axs = plt.subplots(1, self._shelf_number, figsize=figure_size, dpi=dpi)
        if self._shelf_number == 1:
            axs = [axs]  # Convert to list if only one subplot

        for i, ax in enumerate(axs):
            left = i / self._shelf_number + i * 0.05 / (self._shelf_number - 1)
            width = 1 / self._shelf_number - 0.05 / (self._shelf_number - 1)

            ax.set_position([left, 0, width, 1])            
            ax.axis('off')

        return fig, axs


    def convert2img(self):
        self._fig.canvas.draw()

        data = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))

        img = Image.fromarray(data)

        self._data = data
        self._img = img

        return img


    def fig2npa(self, fig):
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def render(self, state):
        self._shelf_number = len(self._objects['shelf'])
        self._can_number = len(self._objects['can'])

        self.states = state
        self._nonfluent_layout = self.build_nonfluents_layout()
        self._state_layout = self.build_states_layout(state)
        self._figure_size = (17*self._shelf_number,22)
        self._fig, self._ax = self.init_canvas(self._figure_size, self._dpi)

        self.plot_state()

        img = self.convert2img()

        for ax in self._ax:
            ax.cla()
        plt.clf()
        plt.close()

        return img
        




        