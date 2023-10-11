import os
import matplotlib.pyplot as plt
import numpy as np


def plot_heatmaps(attention_pattern, show=False, save=True, save_dir='plots', layer='all', head='all', type='attention_pattern'):
   
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    patches = [f'P{i}' for i in range(197)]

    if layer == 'all':
        layer = [i for i in range(12)]
    else:
       layer = [int(layer)]
    
    if head == 'all':
        head = [i for i in range(12)]
    else:
        head = [int(head)]

    for i in range(len(layer)):
            
        for j in range(len(head)):
        
            fig, ax = plt.subplots(figsize=(50,50))
            im = ax.imshow(attention_pattern[layer[i], head[j], :, :], cmap = 'Greys')

            ax.set_yticks(np.arange(len(patches)), labels=patches)
            ax.set_xticks(np.arange(len(patches)), labels=patches)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")

            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Attention', rotation=-90, va="bottom")


            fig.tight_layout()

            if save:
                if type == 'attention_pattern':
                    ax.set_title(f'L{layer[i]}_H{head[j]}: Attention Pattern')
                    plt.savefig(f'{save_dir}/L{layer[i]}_H{head[j]}_attention_pattern.png')
                elif type == 'value_normed':
                    ax.set_title(f'L{layer[i]}_H{head[j]}: Value-normed Attention')
                    plt.savefig(f'{save_dir}/L{layer[i]}_H{head[j]}_value_normed_attention.png')
            elif show:
                plt.show()

            plt.close(fig)