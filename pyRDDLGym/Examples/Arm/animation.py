import matplotlib.pyplot as plt
import os
from PIL import Image
from PIL import Image
import numpy as np

def get_sorted_filenames(directory):
    filenames = os.listdir(directory)
    filenames = [filename for filename in filenames if filename.endswith('.png')]
    filenames = sorted(filenames, key=lambda x: int(x.split('_')[1].split('.')[0]))
    return [os.path.join(directory, filename) for filename in filenames]


def create_video(resolution=(1008, 512), fps=3):
    video_name = 'Examples/Arm/Video/execution.gif'
    directory = 'Examples/Arm/States'

    images = []
    for filename in get_sorted_filenames(directory):
        image = Image.open(filename)
        image = image.resize(resolution)
        image = image.convert('RGB')
        images.append(image)

    images[0].save(video_name, save_all=True, append_images=images[1:], duration=1000/fps, loop=0)


def parse_state(state, index,  can_sizes, shelf_sizes, deter):
    # Count number of shelves in state
    num_shelves = sum(['working-shelf___' in key for key in state.keys()])
    # Create subplots
    fig, axs = plt.subplots(1, num_shelves, figsize=(5*num_shelves,5))
    if num_shelves == 1:
        axs = [axs]

    # Loop over shelves
    for i in range(num_shelves):
        shelf_num = i+1

        #draw shelf
        axs[i].add_patch(plt.Rectangle((shelf_sizes[i][0], shelf_sizes[i][2]), shelf_sizes[i][1], shelf_sizes[i][3], color='#997950'))

        # Plot arm
        arm_x = state['x_position_a']
        arm_y = state['y_position_a']
        if state[f'working-shelf___s{shelf_num}']:
            axs[i].add_patch(plt.Rectangle((arm_x, -6), 1, arm_y+6, color='black'))

        # Get coordinates of cans on shelf
        can_keys = [key for key in state.keys() if (key.startswith('on-shelf___c') and key.endswith(str(shelf_num))) or ('holding___c' in key and state[key] and state[f'working-shelf___s{shelf_num}'])]
        for can_key in can_keys:
            if state[can_key]:
                can_name = can_key.split('__')[1]
                can_x = state[f'x_position_c__{can_name}']
                can_y = state[f'y_position_c__{can_name}']
                if 'holding___c' in can_key:
                    colour = '#DA0016'
                else:
                    colour = '#F40049'
                axs[i].add_patch(plt.Rectangle((can_x, can_y), can_sizes[int(can_name[-1])][0], can_sizes[int(can_name[-1])][1], color=colour))

        # Set axis limits
        axs[i].set_xlim([shelf_sizes[i][0], shelf_sizes[i][1]])
        axs[i].set_ylim([shelf_sizes[i][2]-6, shelf_sizes[i][3]])

    # Show plot
    plt.savefig(f'Examples/Arm/States/{deter+str(index)}.png')
    plt.close()
