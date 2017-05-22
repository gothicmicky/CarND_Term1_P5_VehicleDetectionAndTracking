import bbox
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
import cv2
import glob

bbox = bbox.bbox()
bbox.get_param()
images = glob.glob('test_images/*')
bbox_list = []

# Vehicle detection pipeline
def bbox_pipeline(bbox, img, bbox_list=[]):
    '''
    Processing vehicle detection and bounding box.
    '''
    img = np.copy(img)

    # Do multi-scale searching
    scale = 1.0
    bbox_list = bbox.find_cars(img, scale, bbox_list)
    scale = 1.5
    bbox_list = bbox.find_cars(img, scale, bbox_list)
    scale = 2.0
    bbox_list = bbox.find_cars(img, scale, bbox_list)
    
    ### Heatmap and labelledbounding box
    # Heat map
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = bbox.add_heat(heat,bbox_list)
    # Apply threshold to help remove false positives
    heat = bbox.apply_threshold(heat,10)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Label bounding box
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    # img = bbox.draw_bboxes(img, bbox_list)
    # draw_img = bbox.draw_labeled_bboxes(img, labels)
    
    # To view the heatmap boxes?
    draw_img = np.array(np.dstack((heatmap, heatmap, heatmap))*255, dtype='uint8')
    
    # Alpha blending
    draw_img = cv2.addWeighted(draw_img, 0.5, img, 0.5, 0) 

    # Searching window (big and small)
    s_win = ((bbox.xstart_s,bbox.ystart_s), (bbox.xstop_s,bbox.ystop_s))
    b_win = ((bbox.xstart,bbox.ystart), (bbox.xstop,bbox.ystop))
    cv2.rectangle(draw_img, s_win[0], s_win[1], (255,0,0), 2)
    cv2.rectangle(draw_img, b_win[0], b_win[1], (255,0,0), 2)

    return draw_img

for idx, f in enumerate(images):
	print(f)
	img = mpimg.imread(f)
	print(img.shape)
	res = bbox_pipeline(bbox, img, bbox_list)
	plt.imshow(res)
	# plt.show()
	plt.imsave('output_images/heat_map'+str(idx)+'.jpg',res)
