import imageio
images = []
filenames = []
for i in range(1,50):
    filenames.append('postpic_gif{}.png'.format(i))
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('gif.gif', images, duration = 0.5)
