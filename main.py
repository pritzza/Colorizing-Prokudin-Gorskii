import cv2 as cv
import numpy as np
import skimage.io as skio

def normalize_array(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def crop_borders(img, crop_size=15):
    img_w = img.shape[1]
    img_h = img.shape[0]
    return img[crop_size:img_h-crop_size, crop_size:img_w-crop_size]

def l2_dist(im1, im2):
    return np.sqrt(np.sum(np.square(im1 - im2)))

def shift(img, offset):
    return np.roll(img, shift=offset, axis=(1, 0))

def pyramid_down(im):
    blurred = cv.blur(im, (3, 3))
    downscaled = blurred[::2, ::2]
    return downscaled

def create_pyramid(img, levels):
    pyramid = [img]
    for _ in range(1, levels):
        img = pyramid_down(img)
        pyramid.append(img)
    return pyramid

def compute_edge_magnitude(img):
    sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x**2 + sobel_y**2)
    return normalize_array(mag)

def get_offset(fixed, aligning, search_range=30):
    
    fixed = crop_borders(fixed, fixed.shape[0]//4)
    aligning = crop_borders(aligning, aligning.shape[0]//4)

    fixed_edges = compute_edge_magnitude(fixed)
    aligning_edges = compute_edge_magnitude(aligning)

    min_error = 1000000000
    best_offset = (0, 0)

    for x in range(-search_range//2, search_range//2):
        for y in range(-search_range//2, search_range//2):
            shifted_aligning = shift(aligning_edges, (x, y))

            error = l2_dist(fixed_edges, shifted_aligning)
            
            if error < min_error:
                min_error = error
                best_offset = (x, y)
                
    return best_offset

def pyramid_align(fixed, aligning, levels=4, search_range=30, max_search_area=256):
   
    fixed_pyramid = create_pyramid(fixed, levels)
    aligning_pyramid = create_pyramid(aligning, levels)

    offset = (0, 0)

    for level in range(levels - 1, -1, -1):
        fixed_level = fixed_pyramid[level]
        aligning_level = aligning_pyramid[level]
        
        scaled_offset = offset
        if level < levels - 1:
            scaled_offset = (offset[0] * 2, offset[1] * 2)
        
        # limit to 256x256 area
        search_range = min(search_range, max_search_area // (2 ** level))

        roughly_aligned = shift(aligning_level, scaled_offset)
        refined_offset = get_offset(fixed_level, roughly_aligned, search_range)
        
        offset = (scaled_offset[0] + refined_offset[0], scaled_offset[1] + refined_offset[1])
        print(f"Level {level}: Offset {offset}")

    return offset

def do_the_thing(file_name):
    
    file_path = "data/" + file_name
    img = cv.imread(file_path)
    print("Processing " + file_path)

    img,_,_ = cv.split(img)
    
    height = np.floor(img.shape[0] / 3.0).astype(int)
    width = img.shape[1]

    # Separate color channels
    b = img[:height]
    g = img[height: 2*height]
    r = img[2*height: 3*height]

    b = crop_borders(b, 50)
    g = crop_borders(g, 50)
    r = crop_borders(r, 50)

    red_offset = (0,0)
    green_offset = (0,0)

    if (width * height > 500 * 500):
        red_offset = pyramid_align(b, r, levels=4, search_range=40, max_search_area=256)
        green_offset = pyramid_align(b, g, levels=4, search_range=40, max_search_area=256)
    else:
        red_offset = get_offset(b, r, search_range=30)
        green_offset = get_offset(b, g, search_range=30)
    
    print("red offset", red_offset)
    print("green offset", green_offset)

    aligned_red = shift(r, red_offset)
    aligned_green = shift(g, green_offset)

    # save colored image to disk
    colored_img = cv.merge([b, aligned_green, aligned_red])  # BGR because of opencv
    out_file_name = "results/" + file_name.rsplit('.', 1)[0] + ".jpg"
    cv.imwrite(out_file_name, colored_img)
    print("Saved to " + out_file_name)

    # display colored image at runtime
    #colored_img = cv.merge([aligned_red, aligned_green, b])  # RGB for display
    #skio.imshow(colored_img)
    #skio.show()


files = [
    "tobolsk.jpg", 
    "monastery.jpg", 
    "cathedral.jpg",
    "icon.tif", 
    "lady.tif", 
    "melons.tif", 
    "train.tif",
    "castle.tif",
    "harvesters.tif",
    "onion_church.tif",
    "self_portrait.tif",
    "three_generations.tif",
    "workshop.tif",
    "emir.tif"
]

files = [
    "tobolsk.jpg", 
    "monastery.jpg", 
    "cathedral.jpg",
    "dagestan.tif",
    "lilies.tif",
    "road.tif",
    "creek.tif"
]

for f in files:
    do_the_thing(f)
