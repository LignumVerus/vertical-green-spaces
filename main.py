#run 'Set-ExecutionPolicy Unrestricted -Scope Process' and '.venv\Scripts\activate' to active env
#run 'pipreqs . --ignore ".venv" --force' to create requirements file

import leafmap
import random
from samgeo import tms_to_geotiff
from samgeo.text_sam import LangSAM

from road_network import *
from process_data import *

if __name__ == "__main__":
    #sam = LangSAM()

    # TODO: As inputs. Token cannot be made public
    access_token = 'MLY|6485067431536537|baf4334b590f31d5b78eff1dbd2743f1'
    distance = 50
    num_sample_images = 10
    begin = None
    end = None

    # Choose whether to save the output files
    save = False

    #First we need to load the OpenStreetMap road network to eventually get Mapillary steet view images along this graph.
    city = 'De Uithof, Utrecht'
    prepare_folders(city)

    file_path_features = os.path.join("results", city, "points", "points.gpkg")  
    file_path_road = os.path.join("results", city, "roads", "roads.gpkg") 

    if not os.path.exists(file_path_features):
        # Get the sample points and the features assigned to each point
        road = get_road_network(city)

        # Save road in gpkg file
        road["index"] = road.index
        road["index"] = road["index"].astype(str)
        road["highway"] = road["highway"].astype(str)
        road["length"] = road["length"].astype(float)
        road["road_angle"] = road["road_angle"].astype(float)

        points = select_points_on_road_network(road, distance)

        features = get_features_on_points(points, road, access_token, distance)

        if save:
            road[["index", "geometry", "length", "highway", "road_angle"]].to_file(file_path_road, driver="GPKG", crs=road.crs)
            points.to_csv("testing.csv", index=False)
            features.to_file(file_path_features, driver="GPKG")

        # Set True for n randomly selected rows to analyze their images later
        sample_indices = random.sample(range(len(features)), num_sample_images)
        features["save_sample"] = False
        features.loc[sample_indices, "save_sample"] = True
    else:
        # If the points file already exists, then we use it to continue with the analysis
        features = gpd.read_file(file_path_features, layer="points")

    features = features.sort_values(by='id')

    # If we include a begin and end value, then the dataframe is split and we are going to analyse just those points
    if begin != None and end != None:   
        features = features.iloc[begin:end]

    # Here is where the image downloading and segmentation starts

    # Everything below here is for testing purposes

    header = {'Authorization': 'OAuth {}'.format(access_token)}
    # Get any info from url by concatenating fields like A,B,C
    #https://www.mapillary.com/developer/api-documentation?locale=nl_NL CTRL+F: Fields
    # Image: '468368040925074' | Other images: '2966966740297407' '2052394178235341' '498797444642740'
    url = 'https://graph.mapillary.com/2052394178235341?fields=thumb_original_url,compass_angle'

    # Send a GET request to the Mapillary API to obtain the image URL
    response = requests.get(url, headers=header)
    data = response.json()
    
    # Extract the image URL from the response data
    image_url = data["thumb_original_url"]
    panoramic = True

    #TODO: Voor het gemak nu camera en road angle hetzelfde omdat de camera strak de weg vooruit kijkt
    road_angle = 288
    net_angle = 0

    #Compass angle: the angle from which the photo was taken in relation to true North. For panoramic pictures, this means this angle is in the middle of the picture
    camera_angle = data["compass_angle"]
    print(f"Angle: {camera_angle}")

    new_image = Image.open(requests.get(image_url, stream=True).raw)
    new_image.show()

    sv_images = process_image(image_url, panoramic, road_angle=road_angle)

    for img in sv_images:
        img.show()
