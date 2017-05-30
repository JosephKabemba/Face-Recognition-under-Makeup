######################################################################
# This file is for DTCWT+LBP Face Recognition                        #
######################################################################

################## VMU1 Setting ######################################
print "################## VMU1 Setting ##################"
from setting import *
from CheckPath import *
from CMCData import *
import DTCWT_Processing
txt_dst = "VMU1.txt"
[path, path2, txt_train_dst, txt_test_dst, dst1, dst2] = CheckPathSetting1()
radius = 3
no_points = 8 * radius 
fig_numb = 8


################## VMU1 Label File Name Processing ###################
print "################## VMU1 Label File Name Processing ##################"
train_dic = {}
train_dic2 = {}
test_dic = {}
test_dic2 = {}
values = []
values2 = []

for file in glob.glob(os.path.join(path, "*.mat")):
    name_begin = file.rfind("\\")
    name_end = file.index("d")
    name = file[name_begin+1:name_end]
    #should be 04265

    label_numb = file.rfind("_")
    if label_numb != -1:
        label = file[label_numb+1]
        #shoud be _m
    else:
        label = "original"

    file_name = file[name_begin+1:]
    #"balala.mat"
    dict2_end = file.rfind("for")
    dict2 = file[name_begin+1:dict2_end]
    #shoule be 04265d138_e or 04265d138

    #if label =="m": #y for1 do makeup
    #if label =="l": #y for1 do makeup
    #if label =="e": #y for1 do makeup
    if label !="original": #y for1 do makeup
        shutil.copy(file, dst1)
        f1 = open(txt_test_dst,"a")
        f1.write(file + " " +name+"\n")
        test_dic[file_name] = name
        values.append(name)
        test_dic2[file_name] = dict2
    elif label =="original":#n for 0 do NOT makeup
        shutil.copy(file, dst2)
        f2 = open(txt_train_dst,"a")
        f2.write(file + " " +name+"\n")
        train_dic[file_name] = name
        values2.append(name)
        train_dic2[file_name] = dict2
    else:
        pass

f1.close()
f2.close()
train_images = cvutils.imlist(dst2)
test_ppl_numb = len(test_dic)/fig_numb
train_ppl_numb = len(train_dic)/fig_numb

l2 = sorted(set(values),key=values.index)
l3 = sorted(set(values2),key=values2.index)


################## VMU1 Training Data ################################
print "################## VMU1 Training Data ##################"
X_name = []
y_train = []
tmp = []
pre_label_name = ""
# For each image in the training set calculate the LBP histogram
# and update X_test, X_name and y_test
for train_image in train_images:
    # Read the image
    im_gray_tmp =  sio.loadmat(train_image)
    im_gray = im_gray_tmp.values()[0]
    # Uniform LBP is used
    lbp1_tmp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    shape = len(lbp1_tmp.shape)
    size2 = 1
    for entry2 in range(shape):
        size2 = size2*lbp1_tmp.shape[entry2]
    lbp1 =  np.empty([1, size2])
    lbp1 = np.reshape(lbp1_tmp,-1)
    label_name = train_dic2[os.path.split(train_image)[1]]
    label = train_dic[os.path.split(train_image)[1]]
    
    if pre_label_name == "":
        pre_label_name = label_name
        tmp = lbp1
        X_train = np.zeros((1,fig_numb*tmp.shape[0]))
        y_train.append(label)
    else:
        if pre_label_name == label_name:
            tmp = np.append(tmp, lbp1, axis=0)
        else:
            tmp2 = np.zeros((1,tmp.shape[0]))
            tmp2[0] = tmp
            X_train = np.append(X_train, tmp2, axis=0)
            tmp = lbp1
            pre_label_name = label_name
            y_train.append(label)

tmp2 = np.zeros((1,tmp.shape[0]))
tmp2[0] = tmp
X_train = np.append(X_train, tmp2, axis=0)
X_train = X_train[1:]


################## VMU1 Test Data ####################################
print "################## VMU1 Test Data ##################"
X_name = []
y_test = []
tmp = []
pre_label_name = ""
test_images = cvutils.imlist(dst1)
for test_image in test_images:
    # Read the image
    im_gray_tmp =  sio.loadmat(test_image)
    im_gray = im_gray_tmp.values()[0]
    # Uniform LBP is used
    lbp1_tmp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    
    shape = len(lbp1_tmp.shape)
    size2 = 1
    for entry2 in range(shape):
        size2 = size2*lbp1_tmp.shape[entry2]
    lbp1 =  np.empty([1, size2])
    lbp1 = np.reshape(lbp1_tmp,-1)

    label_name = test_dic2[os.path.split(test_image)[1]]
    label = test_dic[os.path.split(test_image)[1]]

    if pre_label_name == "":
        pre_label_name = label_name
        tmp = lbp1
        X_test = np.zeros((1,fig_numb*tmp.shape[0]))
        y_test.append(label)
    else:
        if pre_label_name == label_name:
                tmp = np.append(tmp, lbp1, axis=0)
        else:
                tmp2 = np.zeros((1,tmp.shape[0]))
                tmp2[0] = tmp
                X_test = np.append(X_test, tmp2, axis=0)
                tmp = lbp1
                pre_label_name = label_name
                y_test.append(label)

tmp2 = np.zeros((1,tmp.shape[0]))
tmp2[0] = tmp
X_test = np.append(X_test, tmp2, axis=0)
X_test = X_test[1:]


################## VMU1 Plot CMC ######################################
print "################## VMU1 Plot CMC ##################"
rank,fpr,tpr,roc_auc = CMC_Data(X_train,y_train,X_test,y_test,l2,l3,txt_dst,data_txt)