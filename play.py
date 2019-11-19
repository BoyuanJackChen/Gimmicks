from toolkit import *
import shutil
import sys
BASE_DIR = back_one_address(sys.path[0])
DATA_DIR = BASE_DIR+'/All_Data'
HAAR_CASCADE_DIR = BASE_DIR+'/haarcascades'

create_d_folder(DATA_DIR+"/Faces_50", "d_Faces_50", verbose=False)


''' get_stats
# delete_unsharp_images(DATA_DIR+'/Eyes_test1', 900, True)
# 第一个图是统计的histogram，第一个图详细列举每一个文件夹里有多少个
# get_stats_file_num(DATA_DIR+'/Batch_data', True, -4, True, 3)
# get_stats_image_size(DATA_DIR+'/Faces')
# '''


''' use cascade
cascade_file = '/outsources/mcs_nose.xml'
cascade_path = HAAR_CASCADE_DIR + cascade_file
create_cropped_folder(DATA_DIR+'/Faces_50', cascade_path, organ_name='Nose_50', 
	verbose=True, scale_Factor=1, min_Neighbors=5)
'''


''' get_big_faces
os.chdir(BASE_DIR)
# delete_few_faces(BASE_DIR+'/Faces', 10)
get_big_faces(BASE_DIR+'/Faces', 40)
'''


''' create_cropped_folder
cascade_file = '/haarcascade_eye.xml'
cascade_path = 'HAAR_CASCADE_DIR' + cascade_file
image_dir = DATA_DIR+'/Faces'
create_cropped_folder(image_dir, cascade_path, "Eyes", True)
'''


''' Paint colorful graph
graph_address = DATA_DIR + '/alt2_face_cascade.png'
draw_colorful_graph(graph_address, stride=5)

d_graph = get_image_d_absolute(graph_address)
draw_colorful_graph(d_graph, 1, "Image")

dd_graph = get_image_d_absolute(d_graph,"Image")
draw_colorful_graph(dd_graph, 1, "Image")
#'''


''' Test Stanford code
base_dir = back_one_address(sys.path[0])
print("Your base_dir is:\n", base_dir)
im.tester.stupid()

x_train, y_train = load_files(base_dir+'Faces', verbose=True)
x_val, y_val = load_files(base_dir+'Faces_test', verbose=True)

model = im.fc_net.FullyConnectedNet([100, 100],
              weight_scale=1e-3, dtype=np.float64)
data = {'x_train':x_trian, 'y_train':y_train, }
solver = im.solver.Solver(model, data, lr_decay=0.95, optim_config={'learning_rate':1e-3}, 
                print_every = 100, verbose=True)
'''


''' Draw Putin's left eye; test get_spiral
graph_address = '../Putin/Left_Eye.png'
image = cv2.imread(graph_address)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

draw_colorful_graph(graph_address, 1)

d_graph = get_image_d_absolute(graph_address)
draw_colorful_graph(d_graph, 1, "a")

r,g = get_spiral(gray)
for (c,x,y,z) in r:
	print(x,y)
draw_colorful_graph(g, 1,'a')
'''