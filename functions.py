from modules import *

def load_image_polygon_from_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    image = data['image']
    imbytes = BytesIO(base64.b64decode(image))
    image = Image.open(imbytes)
    image = image.convert("RGB")
    image = np.array(image)
    image = sk.img_as_float32(image)
    polygon = data['polygon']
    return image, polygon

def draw_boundary(image, polygon):
    boundary_image = np.copy(image)
    cv2.polylines(boundary_image,[np.array(polygon[0])],True,(1,0,0), thickness = 1, lineType = cv2.LINE_AA)
    return boundary_image

def save_all_boundary_images(data_dir):
    json_filenames = np.array([data_dir + x for x in os.listdir(data_dir) if x.endswith(".JSON")])
    for json_filename in json_filenames:
        image, polygon = load_image_polygon_from_json(json_filename)
        boundary_image = draw_boundary(image, polygon)
        jpeg_filename = json_filename.replace('.JSON', '.jpg')
        boundary_image = sk.img_as_ubyte(boundary_image)
        io.imsave(jpeg_filename, boundary_image)

def save_selected_fields_geojson(filename_geojson, dir_selected_fields, name_savefile):
  selected_ids = [x.replace('.JSON', '') for x in os.listdir(dir_selected_fields) if x.endswith(".JSON")]
  with open(filename_geojson) as json_file:
    data = json.load(json_file)
  features = data['features']
  selected_features = []
  for i in range(0, len(features)):
    current_id = features[i]['properties']['id']
    if current_id in selected_ids:
      selected_features.append(features[i])
  #print(features)
  data['features'] = selected_features
  with open(name_savefile, 'w') as outfile:
    json.dump(data, outfile)
  return data

def remove_rejeted_fields(data_dir):
  json_filenames = [data_dir + x for x in os.listdir(data_dir) if x.endswith(".JSON")]
  jpg_filenames = [data_dir + x for x in os.listdir(data_dir) if x.endswith(".jpg")]
  for json_filename in json_filenames:
    jpg_filename = json_filename.replace('.JSON', '.jpg')
    if jpg_filename not in jpg_filenames:
      if os.path.exists(json_filename):
        os.remove(json_filename)