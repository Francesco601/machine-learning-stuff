from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont


# Load font
font = ImageFont.truetype("arial.ttf", 40)

# Initialize the object detection pipeline
object_detector = pipeline("object-detection")


# Draw bounding box definition
def draw_bounding_box(im, score, label, xmin, ymin, xmax, ymax, index, num_boxes):
	""" Draw a bounding box. """

	print(f"Drawing bounding box {index} of {num_boxes}...")

	# Draw the actual bounding box
	im_with_rectangle = ImageDraw.Draw(im)  
	im_with_rectangle.rounded_rectangle((xmin, ymin, xmax, ymax), outline = "red", width = 5, radius = 10)

	# Draw the label
	im_with_rectangle.text((xmin+35, ymin-25), label, fill="white", stroke_fill = "red", font = font)

	# Return the intermediate result
	return im


# Open the image
with Image.open("street.jpg") as im:

	# Perform object detection
	bounding_boxes = object_detector(im)

	# Iteration elements
	num_boxes = len(bounding_boxes)
	index = 0

	# Draw bounding box for each result
	for bounding_box in bounding_boxes:

		# Get actual box
		box = bounding_box["box"]

		# Draw the bounding box
		im = draw_bounding_box(im, bounding_box["score"], bounding_box["label"],\
			box["xmin"], box["ymin"], box["xmax"], box["ymax"], index, num_boxes)

		# Increase index by one
		index += 1

	# Save image
	im.save("street_bboxes.jpg")

	# Done
	print("Done!")


        
