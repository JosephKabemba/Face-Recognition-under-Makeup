import setting

predictor_path = "shape_predictor_68_face_landmarks.dat"
faces_folder_path = ".\\Origion\VMU"
path = ".\\Post\W\VMU"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

win = dlib.image_window()
for f in glob.glob(os.path.join(faces_folder_path, "*.bmp")):
    img = io.imread(f)
    win.clear_overlay()
    win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #    k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        
        # Draw the face landmarks on the screen.
        win.add_overlay(shape)

        list = []
        for i in range(68):
            pt = shape.part(i)
            list.append(pt)
        
        point_list = []
        for entry0 in list:
            point_list.append([entry0.x,entry0.y])
        point_list_fin = np.array(point_list)
        hull = ConvexHull(point_list_fin)

        # Get the indices of the hull points.
        hull_indices = hull.vertices
        mask = np.zeros(img.shape, dtype=np.uint8)

        # These are the actual points.
        hull_pts = point_list_fin[hull_indices, :]
        roi_corners = np.array([hull_pts],dtype=np.int32)
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image

        ignore_mask_color = (255,)*channel_count
        ignore_mask_color2 = (0,)*channel_count
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)

        # get masked foreground
        masked_image = cv2.bitwise_and(img, mask)
        mask2 = cv2.bitwise_not(mask)
        final = cv2.bitwise_or(masked_image, mask2)

        # save the result
        from PIL import Image
        im = Image.fromarray(final)
        name_begin = f.rfind("\\")
        name_end = f.index(".bmp")
        name_tmp = f[name_begin+1:name_end]
        path2 = os.path.join(path, str(name_tmp))
        name = path2 + ".bmp"
        im.save(name)      

    win.add_overlay(dets)
    #dlib.hit_enter_to_continue()