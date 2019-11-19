import cv2
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image    # pillow, installed from pip
import pickle
import math
import argparse
import shutil
import heapq
import sys
from sys import platform
DARWIN = (platform == "darwin")
slash = '/' if DARWIN else '\\'

''' Menu & Usage

create_d_folder(image_dir, outdir_name, verbose=True)
get_image_d_absolute(image_path, type="string")

get_stats_file_num(image_dir, num_best_values=10, need_last=False, last=-3, verbose=False, step_size=10)
get_stats_image_size(image_dir)
get_big_faces(image_dir, num_threshold=40, verbose=True, num_people=0)

get_spiral(gray)
find_center(image, is_gray="False")

draw_colorful_graph(image_path, stride=3, type="string")

delete_few_faces(image_dir, num_threshold=40, verbose=True)
	- This is a direct delete. Make sure you have saved the copy. 

create_cropped_folder(image_dir, cascade_file_address, organ_name='Faces', verbose=False
	, scale_Factor=1, min_Neighbors=5)
draw_organ(image_address, cascade_file_address, verbose=False, scale_Factor=1, min_Neighbors=5)
	Usage:
	- image_dir: Image folder with respect to the location of .py file
	- cascade_file_address: .xml file in haarcascade folder
	- organ_name: "Faces", "Mouths", sth like that
	- verbose: If you would like to print the progress in terminal
	- scale_Factor: haarCascade parameter
	- mn_Neighbors: haarCascade parameter

count_files(dir_address)
back_one_address(address)    
clean_file_name(address)
get_file_name(address)
last_dir_name(address)

load_files_w_face_cascade(image_dir, cascade_file_add, verbose=False)
load_files(image_dir, verbose=False)
smart_3D_reshape
-- Now they still resize it for you

'''

def get_spiral(gray, center=None):
	if (center==None):
		O = find_center(gray, True)
	else:
		O = center
	x = O[0]
	y = O[1]
	counter = 0
	step = 0
	inc_step = True
	result = []
	result.append(np.array([counter, x, y, gray[x][y]]))
	direction = ["D", "R", "U", "L"]
	dir_i = 0
	while ((x >= 0) and x < gray.shape[0] and y >= 0 and y < gray.shape[1]):
		if (inc_step):
			step += 1
		inc_step = not(inc_step)
		for i in range(0,step):
			if (dir_i == 0):
				y-=1
			elif (dir_i==1):
				x+=1
			elif (dir_i==2):
				y+=1
			else:
				x-=1
			counter+=1
			if ((x >= 0) and x < gray.shape[0] and y >= 0 and y < gray.shape[1]):
				result.append(np.array([counter, x, y, gray[x][y]]))
			else:
				break
		dir_i+= 1
		if (dir_i == 4):
			dir_i = 0
	the_graph = np.zeros(gray.shape)
	for c, i, j, z in result:
		the_graph[i][j] += z
	the_graph[O[0], O[1]] = -100
	result = np.array(result)
	return (result, the_graph)


def find_center(img, is_gray = "False", thresh=-1): 
	if not is_gray:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		gray = img
	if (thresh>=0): 
		ret,thresh = cv2.threshold(gray,70,255,0)
		thresh = cv2.erode(thresh, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=2)
		thresh = 255 - thresh
        # calculate moments of binary image
		M = cv2.moments(thresh)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
	else:
		gray = 255-gray
		M = cv2.moments(gray)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
	return np.array([cX, cY])

# Not converting well. For some reason it only saves to a black image.
def create_d_folder(image_dir, outdir_name, verbose=True):
	current_dir = os.getcwd()
	parent_dir = back_one_address(image_dir)
	os.chdir(parent_dir)
	if not os.path.exists(outdir_name):
		os.makedirs(outdir_name)
	eye_dir = parent_dir + slash + outdir_name
	label_ids = []      # A list of dir names

	for root, dirs, files in os.walk(image_dir):
		for file in files:
			file_name = get_file_name(root)
			if (file_name==get_file_name(image_dir)) or (file.endswith('.DS_Store')):
				continue
			if not file_name in label_ids:
				os.chdir(eye_dir)
				if not os.path.exists(file_name):
					os.makedirs(file_name)
				os.chdir(eye_dir+slash+file_name)
			full_address = root+slash+file
			dW_graph = get_image_d_absolute(full_address)
			for i in range(0,dW_graph.shape[0]):
				for j in range(0,dW_graph.shape[1]):
					dW_graph[i][j] = int(dW_graph[i][j])

			if file=="0001_01.png":
				print(f"dW_graph.shape was {dW_graph.shape}")
				im = Image.fromarray(dW_graph)
				im.show()

			dW_graph = np.reshape(dW_graph, (dW_graph.shape[0],dW_graph.shape[1],1))
			if file=="0001_01.png":
				print(f"dW_graph.shape is {dW_graph.shape}; \n {dW_graph}")
			cv2.imwrite(file, dW_graph)
			if verbose:
				print(f"Finished generating dW image for {file}")

	os.chdir(current_dir)
	return


def get_image_d_absolute(image_path, type="string", to_int=True):
	# If the parameter is a path, read the path; if it is already a gray image, use it.
	if type=="string":
		image = cv2.imread(image_path)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	elif(len(image_path.shape)==2):
		gray = image_path
	else:
		gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
	gray = np.array(gray)
	result = np.zeros(gray.shape)
	for m in range(0,gray.shape[0]):
		for n in range(0,gray.shape[1]):
			if (m==gray.shape[0]-1 or m==gray.shape[0]):
				dm=0
			else:
				dm=gray[m+1][n]-gray[m][n]
			if (n==gray.shape[1]-1 or n==gray.shape[1] or n==gray.shape[1]-2):
				dn=0
			else:
				dn=gray[m][n+1]-gray[m][n]
			sqrt = math.sqrt(dm*dm+dn*dn)
			if to_int:
				sqrt = int(sqrt)
			result[m][n]+=sqrt
	if to_int:
		result.astype("uint8")
	return result


def draw_colorful_graph(image_path, stride=3, type="string"):
	if type=="string":
		image = cv2.imread(image_path)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	else:
		gray = image_path

	gray = 255-gray
	gray = np.array(gray)

	x = np.arange(len(gray))
	y = np.arange(len(gray[0]))
	X, Y = np.meshgrid(x, y)

	plt.figure(1)
	ax = plt.axes(projection='3d')
	ax.plot_surface(X, Y, np.transpose(gray), rstride=stride, cstride=stride,
                cmap='plasma', edgecolor='none')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.view_init(90, 0)
	plt.show()
	return


def delete_unsharp_images(image_dir, threshold, verbose=False):
	# Save current working directory before going to the parent one. 
	current_dir = os.getcwd()
	os.chdir(image_dir)
	
	for root, dirs, files in os.walk(image_dir):
		if (root==image_dir):
			continue
		for file in files:
			if (file.endswith(".DS_Store")):
				continue
			print('root is:',root)
			print('file is:',file)
			image = cv2.imread(root+'/'+file)
			size = len(image) * len(image[0])
			if (size < threshold):
				os.remove(root+slash+file)
			if verbose:
				print("Deleted", file, 'from', get_file_name(root))
	os.chdir(current_dir)
	return


def get_big_faces(image_dir, num_threshold=40, verbose=True, num_people=0):
	current_dir = os.getcwd()
	parent_dir = back_one_address(image_dir)
	os.chdir(parent_dir)
	if not os.path.exists('big_faces'):
		os.makedirs('big_faces')
	os.chdir('big_faces')
	people_names = []
	acc_people=0
	while (num_people<=0 or (acc_people<num_people)):
		for root, dirs, files in os.walk(image_dir):
			if (root==image_dir):
				continue
			if (len(files) >= num_threshold):
				if not root in people_names:
					people_names.append(root)
					os.makedirs(get_file_name(root))
				os.chdir(get_file_name(root))
				for file in files:
					image = cv2.imread(root+'/'+file)
					cv2.imwrite(file, image)
				os.chdir('..')
				acc_people += 1
				if verbose:
					print("Got face of", get_file_name(root))
	os.chdir(current_dir)
	return


def get_stats_image_size(image_dir):
	current_dir = os.getcwd()
	os.chdir(image_dir)
	sizes=[]
	for root, dirs, files in os.walk(image_dir):
		if (root==image_dir):
			continue
		for file in files:
			if (file.endswith(".DS_Store")):
				continue
			image = cv2.imread(root+slash+file)
			size = len(image) * len(image[0])
			sizes.append(size)
	npsizes = np.array(sizes)
	average = np.average(npsizes)
	print("The average size is:", average, '; the square root is', math.sqrt(average))
	os.chdir(current_dir)
	return (average, sizes)
	'''
	num_bins = 10
	plt.title("Size stats for "+get_file_name(image_dir) + " folder")
	plt.hist(sizes, num_bins, facecolor='blue', edgecolor='k', alpha=0.5)
	plt.axvline(average, color='k', linestyle='dashed', linewidth=1)
	plt.xlabel("image size")
	plt.ylabel("Number of files")
	plt.show()
	os.chdir(current_dir)
	'''


def get_stats_file_num(image_dir, num_best_values=10, need_last=False, last=-3, verbose=False, step_size=10):
	current_dir = os.getcwd()
	os.chdir(image_dir)
	total_dir_num = len(os.listdir(image_dir))
	nums=np.array([])
	names=np.array([])
	count=0
	for root, dirs, files in os.walk(image_dir):
		if (root==image_dir):
			continue
		name = get_file_name(root)
		if need_last:
			name = name[last:]
		names = np.append(names, name)
		nums = np.append(nums, len(files))
		count+=1
		if (verbose and count%step_size==0):
			print("Progress: ", count, '/', total_dir_num)
	best_indices = nums.argsort()[:num_best_values:-1]
	best_names = []
	for index in best_indices:
		best_names.append(names[index])
	print("There are", int(np.sum(nums)), "files in", total_dir_num, "folders. On average, each folder has", int(np.average(nums)),"files")
	print("The biggest folders are:", best_names)
	print("They each have", nums[best_indices], "files")
	'''
	fig, axs = plt.subplots(2)
	fig.suptitle('File num stats for '+get_file_name(image_dir)+' folder')
	num_bins = 10
	axs[0].hist(nums, num_bins, facecolor='blue', edgecolor='k', alpha=0.5)
	axs[1].bar(names, nums)
	axs[1].set_xlabel("subfolder names")
	axs[1].set_ylabel("number of files")
	plt.show()
	'''
	os.chdir(current_dir)
	return best_names


def delete_few_faces(image_dir, num_threshold=40, verbose=True):
	current_dir = os.getcwd()
	os.chdir(image_dir)
	
	for root, dirs, files in os.walk(image_dir):
		if (root==image_dir):
			continue
		if (len(files) < num_threshold):
			shutil.rmtree(root)
			if verbose:
				print("Deleted", get_file_name(root))
	os.chdir(current_dir)
	return


def create_cropped_folder(image_dir, cascade_file_address, organ_name, 
	verbose=True, scale_Factor=1, min_Neighbors=5):
	current_dir = os.getcwd()
	parent_dir = back_one_address(image_dir)
	os.chdir(parent_dir)
	if not os.path.exists(organ_name):
		os.makedirs(organ_name)
	eye_dir = parent_dir + slash + organ_name
	label_ids = []      # A list of dir names

	for root, dirs, files in os.walk(image_dir):
		for file in files:
			file_name = get_file_name(root)
			if (file_name==get_file_name(image_dir)) or (file.endswith('.DS_Store')):
				continue
			if not file_name in label_ids:
				os.chdir(eye_dir)
				if not os.path.exists(file_name):
					os.makedirs(file_name)
				os.chdir(eye_dir+slash+file_name)
			full_address = root+slash+file
			draw_organ(full_address, cascade_file_address, verbose)
	os.chdir(current_dir)
	return


def draw_organ(image_address, cascade_file_address, verbose=True, scaleFactor=1.1, minNeighbors=5):
	image = cv2.imread(image_address)
	gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	organ_cascade = cv2.CascadeClassifier(cascade_file_address)
	organs = organ_cascade.detectMultiScale(image, scaleFactor, minNeighbors)
	for (x, y, w, h) in organs:
		roi_color = image[y:y+h, x:x+w]
		new_image_name = clean_file_name(image_address)+'.png'
		cv2.imwrite(new_image_name, roi_color)
		if verbose:
			print("Finished writing organ for ", new_image_name)
	return


def back_one_address(address):
    i = -1
    if (platform == "darwin"):
    	while (address[i]!='/'):
        	i-=1
    else:
        while (address[i]!='\\'):
            i-=1
    return(address[0:i])

def clean_file_name(address):
	result = get_file_name(address)
	i = -1
	while (address[i]!='.'):
		i-=1
	return(result[:i])

def get_file_name(address):
	result = ""
	i = -1
	if (platform == "darwin"):
		while (address[i]!='/'):
			result += address[i]
			i-=1
	else: 
		while (address[i]!='\\'):
			result += address[i]
			i-=1
	return(result[::-1])

def last_dir_name(address):
	result = get_file_name(back_one_address(address))
	return result


# Modified from faces-train.py
def load_files_w_face_cascade(image_dir, cascade_file_address, verbose=False):
	image_dir = os.path.dirname(image_dir)

	face_cascade = cv2.CascadeClassifier(cascade_file_address)

	current_id = 0
	label_ids = {}  # A dictionary of "label" and id_
	y_labels = []
	x_train = []

	for root, dirs, files in os.walk(image_dir):
		for file in files:
			if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
				path = os.path.join(root, file)
				# basename means the string after the last /
				label = os.path.basename(root).replace(" ", "-").lower()
				# Give an id_ to this label. If it doesn't exist, create a new one, refresh current_id
				if not label in label_ids:
					label_ids[label] = current_id
					current_id += 1
				id_ = label_ids[label]
				# Fill in the gray image and the id_
				pil_image = Image.open(path).convert("L") # grayscale
				# Resize the pictures! 
				size = (550, 550)
				final_image = pil_image.resize(size, Image.ANTIALIAS)
				image_array = np.array(final_image, "uint8")
				faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

				# Important: we must not have multiple faces in any image. The code can't deal with that
				for (x,y,w,h) in faces:
					roi = image_array[y:y+h, x:x+w]
					x_train.append(roi)
					y_labels.append(id_)
	print("Loading with face cascade finished. We have ", len(x_train), " images loaded. ")
	return (x_train, y_labels)


def load_files(image_dir, verbose=False):
	current_id = 0
	label_ids = {}  # A dictionary of "label" and id_
	y_labels = []
	x_train = []

	for foldername in os.listdir(image_dir):
		if foldername.endswith(".DS_Store"):
			continue
		file_dir = image_dir + '/' + foldername
		for file in os.listdir(file_dir):
			if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
				path = file_dir + '/' + file
				# basename means the string after the last /
				label = foldername.replace(" ", "-").lower()
				# Give an id_ to this label. If it doesn't exist, create a new one, refresh current_id
				if not label in label_ids:
					label_ids[label] = current_id
					current_id += 1
				id_ = label_ids[label]
				# Fill in the gray image and the id_
				pil_image = Image.open(path)
				# Resize the pictures! 
				size = (100, 100)
				final_image = pil_image.resize(size, Image.ANTIALIAS)
				image_array = np.array(final_image, "uint8")
				image_array = smart_3d_reshape(image_array)
				# Important: we must not have multiple faces in any image. The code can't deal with that
				x_train.append(image_array)
				y_labels.append(id_)
	x_train = np.array(x_train)
	y_labels = np.array(y_labels)
	if (verbose):
		print("image_dir is:", image_dir)
		print("x.shape is:", x_train.shape)
		print("y.shape is:", y_labels.shape)
	return (x_train, y_labels)


def smart_3d_reshape(np_matrix):
	result = []
	for k in range(0,np_matrix.shape[2]):
		sub = np.zeros([np_matrix.shape[0], np_matrix.shape[1]])
		for i in range(0,np_matrix.shape[0]):
			for j in range(0,np_matrix.shape[1]):
				sub[i][j] += np_matrix[i][j][k]
		result.append(sub)
	result = np.array(result)
	return result


def load_files_old(image_dir, verbose=False):
	print("image_dir is: ", image_dir)

	current_id = 0
	label_ids = {}  # A dictionary of "label" and id_
	y_labels = []
	x_train = []

	for foldername in os.listdir(image_dir):
		if foldername.endswith(".DS_Store"):
			continue
		file_dir = image_dir + '/' + foldername
		for file in os.listdir(file_dir):
			if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
				path = file_dir + '/' + file
				# basename means the string after the last /
				label = foldername.replace(" ", "-").lower()
				# Give an id_ to this label. If it doesn't exist, create a new one, refresh current_id
				if not label in label_ids:
					label_ids[label] = current_id
					current_id += 1
				id_ = label_ids[label]
				# Fill in the gray image and the id_
				# pil_image = Image.open(path).convert("L") # grayscale
				pil_image = Image.open(path)
				# Resize the pictures! 
				size = (100, 100)
				final_image = pil_image.resize(size, Image.ANTIALIAS)
				image_array = np.array(final_image, "uint8")

				# Important: we must not have multiple faces in any image. The code can't deal with that
				x_train.append(image_array)
				y_labels.append(id_)

	x_train = np.array(x_train)
	a, b, c, d = x_train.shape
	x_train = x_train.reshape((d, c, b, a))
	print(x_train.shape)
	y_labels = np.array(y_labels)
	print(y_labels.shape)
	return (x_train, y_labels)


def count_files(dir_address):
	current_dir = os.getcwd()
	result = 0
	os.chdir(dir_address)
	for root, dirs, files in os.walk(os.getcwd()):
		for file in files:
			result += 1
	os.chdir(current_dir)
	return result