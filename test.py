import matplotlib.pyplot as plt

def onclick(event):
    print('x:', event.xdata, ' y:', event.ydata)

img = plt.imread('/mnt/B-SSD/maltamed/datasets/2D/ISIC/images/ISIC_0010231.jpg')
fig = plt.figure()
plt.imshow(img)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
