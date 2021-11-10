from modules import *
from unet import *
app = Flask(__name__)

def get_model():
    global model
    dir_images_JSON = 'DATA/ALL_POLYGONS'
    log_dir = "model"
    pretrained_weights = log_dir+'/'+'best_weights.hdf5'
    unet = Unet(dir_images_JSON, 
            input_shape=(256,256,3), 
            pretrained_weights=pretrained_weights, 
            log_dir=log_dir, 
            unet_type="small", 
            filter_size=3, 
            n_filters=16, 
            lr_rate=1e-3)
    model = unet.model

def create_mask_clickpoint(image, clickpoint):
    mask_clickpoint = np.copy(image)
    mask_clickpoint[:,:,:] = 0
    cv2.circle(mask_clickpoint, clickpoint, 3, (1,1,1), thickness=-1, lineType = cv2.LINE_AA)
    mask_clickpoint = mask_clickpoint[:,:,0]
    mask_clickpoint = np.expand_dims(mask_clickpoint, axis=-1)
    mask_clickpoint = np.expand_dims(mask_clickpoint, axis=0)
    return mask_clickpoint
    
def scale_point(point, input_shape=(256,256), output_shape=(600,600)):
    x = point[0]
    y = point[1]
    x_scaled = (x/(input_shape[1]))*output_shape[1]
    x_scaled = int(x_scaled)
    if x_scaled > (output_shape[1] -1):
        x_scaled = output_shape[1] -1
    if x_scaled < 0:
        x_scaled = 0
    y_scaled = (y/(input_shape[0]))*output_shape[0]
    y_scaled = int(y_scaled)
    if y_scaled > (output_shape[0] -1):
        y_scaled = output_shape[0] -1
    if y_scaled < 0:
        y_scaled = 0
    point_scaled = (x_scaled,y_scaled)
    return point_scaled

get_model()

@app.route('/_receive_clickpoint', methods=["POST"])
def receive_clickpoint():
    x = request.get_json(force=True)['x']
    y = request.get_json(force=True)['y']
    x = np.int(x)
    y = np.int(y)
    img = request.get_json(force=True)['img']
    img = img + "="
    img = img[22:]
    imbytes = BytesIO(base64.b64decode(img))
    image = Image.open(imbytes)
    image = image.convert("RGB")
    image = np.array(image)
    image = sk.img_as_float32(image)
    browser_image_shape = (image.shape[0], image.shape[1])
    unet_input_shape = (256,256)
    image = resize(image, unet_input_shape)

    clickpoint = scale_point((x,y), browser_image_shape, unet_input_shape)
    x = clickpoint[0]
    y = clickpoint[1]
    mask_clickpoint = create_mask_clickpoint(image, (x,y))
    seed = mask_clickpoint[0,:,:,0]
    image = np.expand_dims(image, axis=0)

    # Hier findet die Vorhersage statt
    pred = model.predict([image, mask_clickpoint])
    # Ab hier wird das Ergebnis aufbereitet
    image = image[0,:,:,:]
    pred_contour = pred[0,:,:,0]
    indices_pred_contour = np.where(pred_contour >= 0.5)
    pred_contour[indices_pred_contour] = 1
    label_pred_contour, num_contours = measure.label(pred_contour, return_num=True)
    value_at_clickpoint = label_pred_contour[y,x]

    for i in range(1, num_contours+1):
        if i != value_at_clickpoint:
              label_pred_contour[label_pred_contour==i] = 0

    label_pred_contour[label_pred_contour>0] = 1
    pred_contour = label_pred_contour
    pred_contour = ndimage.binary_fill_holes(pred_contour)
    indices_pred_contour = np.where(pred_contour > 0)
    image[:,:,0][indices_pred_contour] = 1
    
    boundaries, hierarchy = cv2.findContours(img_as_ubyte(pred_contour),
                            cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    boundary = boundaries[0]
    cnt = boundary
    epsilon = 0.01*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    boundary = np.squeeze(approx)
    b1 = np.copy(boundary[:,0])
    b2 = np.copy(boundary[:,1])
    boundary = boundary.tolist()
    for i, point in enumerate(boundary):
        point_scaled = scale_point(point, unet_input_shape, browser_image_shape)
        boundary[i] = point_scaled
    return jsonify(result={"ic":boundary})
