from PIL import Image, ImageFilter

image = Image.open('red-panda.jpeg')
image = image.filter(ImageFilter.FIND_EDGES)
image.save('red-panda-edge.jpeg')