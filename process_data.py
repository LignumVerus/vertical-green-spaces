# Functions courtesy of iabrilvzqz and labib (2023)
import os
import requests
from PIL import Image, ImageFile

def prepare_folders(city):
    # Create folder for storing sample points and road network if they don't exist yet

    dir_path = os.path.join("results", city, "points")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.join("results", city, "roads")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    dir_path = os.path.join("results", city, "sample_images")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# TODO: NOT FINISHED YET. Processes a SV image based on an image URL.
# Returns a list of processed images (1 for normal, multiple in case of panoramic)
def process_image(image_url, is_panoramic, road_angle):
    images = []
    #TODO: Crash maybe because everything is in the try statement?
    # try:
    # Fetch and process the image
    image = Image.open(requests.get(image_url, stream=True).raw)

    if is_panoramic:
        width, height = image.size

        # Also crop out bottom 20%
        cropped_height = int(height*0.8)
        image = image.crop((0, 0, width, cropped_height))

        # Generate two perpendicular looking images
        #TODO: Enters the except clause when enabled for some reason
        left_face, right_face = get_perpendicular_images(image, road_angle)

        #TODO: Which of 4 slices are closest to -90 and +90 degrees?

        print("type:")
        print(type(left_face))

        #left face: 1-3rd eighth. right face: 5-7 eighth

        #eighth_width = int(0.125 * width)
        # left_face = restitched_img.crop((eighth_width, 0, eighth_width*3, cropped_height))
        # right_face = restitched_img.crop((eighth_width*5, 0, eighth_width*7, cropped_height))

        #TODO: add defisheye from PyPI?

        images.extend([left_face, right_face])
    else:
        images.append(image)

    return images
    # except:
    #     print("Error downloading image")

def get_perpendicular_images(image, road_angle):
    width, height = image.size
    eighth_width = int(0.125 * width)

    # We want perpendicular left and right facing images. Wrap around in case values are out of bounds
    wanted_angles = ((road_angle - 90) % 360, (road_angle + 90) % 360)

    faces = []
    original_image = image.copy()

    # We want 1/8th of the image before and after the wanted angle within the shot (1/4th total)
    for wanted_angle in wanted_angles:
        image = original_image.copy()

        # E.g. if wanted_angle is 10, the wanted shift is to fraction 0.0278  of the image on a 0-1 range
        wanted_fractional_axis = float(wanted_angle)/360.0

        #TODO: add exception when wanted_axis is greater than image width or when wanted_absolute_shift - eighth_width is less than 0 etc.
        wanted_axis= int(width * wanted_fractional_axis)

        # If the face is partially out of bounds, the image needs to be restitched to allow for everything to be included
        # Use modulo operator
        # if wanted_axis - eighth_width < 0:
        #     # If it's smaller, we want the part sticking out from the right side
        #     pass
        # elif wanted_axis + eighth_width >= 360:
        #     # If it's bigger, we want the part sticking out from the left side
        #     pass

        left_max = max(wanted_axis - eighth_width, 0)
        right_max = min(wanted_axis + eighth_width, width)
        print(left_max, right_max)
        perpendicular_face = image.crop((left_max, 0, right_max, height))

        faces.append(perpendicular_face)

    # Return the left and right perpendicular face
    return faces

def restitch_image(image):
    #cut it into 4 pieces and save only the side angles
    #use PIL image.size, image.format, image.crop
    #assuming panoramas are 360 degrees, we know we want to crop 50% of the image out
    width, height = image.size

    # Restitch the image to make the middle the front facing angle
    # This is done by moving 25% from the right to the left side
    right_width = int(0.25 * width)

    left_img = image.crop((0, 0, width - right_width, height))
    right_img = image.crop((width - right_width, 0, width, height))

    restitched_img = Image.new('RGB', (width, height))
    restitched_img.paste(right_img, (0,0))
    restitched_img.paste(left_img, (right_width, 0))

    return restitched_img

#TODO: NOT FINISHED YET
def download_image(id, geometry, image_id, is_panoramic, save_sample, city, access_token, processor, model):
    # Check if the image id exists
    if image_id:
        try:
            # Create the authorization header for the Mapillary API request
            header = {'Authorization': 'OAuth {}'.format(access_token)}

            # Build the URL to fetch the image thumbnail's original URL
            url = 'https://graph.mapillary.com/{}?fields=thumb_original_url'.format(image_id)
            
            # Send a GET request to the Mapillary API to obtain the image URL
            response = requests.get(url, headers=header)
            data = response.json()
            
            # Extract the image URL from the response data
            image_url = data["thumb_original_url"]

            # Process the downloaded image using the provided image URL, is_panoramic flag, processor, and model
            images, segmentations, result = process_image(image_url, is_panoramic, processor, model)

            if save_sample:
                save_images(city, id, images, segmentations, result[0])

        except:
            # An error occurred during the downloading of the image
            result = [None, None, True, True]
    else:
        # The point doesn't have an associated image, so we set the missing value flags
        result = [None, None, True, False]

    # Insert the coordinates (x and y) and the point ID at the beginning of the result list
    # This helps us associate the values in the result list with their corresponding point
    result.insert(0, geometry.y)
    result.insert(0, geometry.x)
    result.insert(0, id)

    return result