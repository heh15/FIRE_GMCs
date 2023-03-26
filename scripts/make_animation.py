import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np

figDir = '../figures/'
picDir = '../snapshots/'

orbit = 'e2'; incl = 'v0'
snapshotsPart = np.arange(520, 712, 2)

my_dpi = 96
figsize = (588/96,448/96)

fig = plt.figure(figsize=figsize)
ax = plt.subplot(111)

def animate(i):
    ax.clear()
    ax.axis('off')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    idNo = snapshotsPart[i]
    image=plt.imread(picDir+'G2G3_e2_'+str(incl)+'_'+str(idNo)+'.png')
    ax.imshow(image)

anim = FuncAnimation(fig, animate, frames=96, interval=100, repeat=False)
anim_html = HTML(anim.to_jshtml())
with open(figDir+'G2G3_'+orbit+'_'+incl+'_final.html', 'w') as f:
    f.write(anim_html.data)
