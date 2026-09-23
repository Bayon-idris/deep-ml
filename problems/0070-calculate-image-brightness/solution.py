
def calculate_brightness(img):
	if not img or len(img) == 0 or len(img[0]) == 0:
		return -1

	height = len(img)
	width = len(img[0])
	sum = 0
	num_elements = 0

	for y in range(height):
		if len(img[y]) != width:
			return -1

		for x in range(width) : 
			pixel = img[y][x]
			if(pixel <0 or pixel > 255) :
				return -1

			sum += pixel
			num_elements +=1

	return sum/num_elements 
