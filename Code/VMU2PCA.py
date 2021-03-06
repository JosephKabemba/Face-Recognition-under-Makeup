######################################################################
# This file is for PCA Eigenface Recognition.                        #
######################################################################

################## VMU2 Setting ######################################
print "################## VMU2 Setting ##################"
from setting import *
from CheckPath import *
from CMCData import *
txt_dst = "VMU2.txt"
[path, path2, txt_train_dst, txt_test_dst, dst1, dst2] = CheckPathSetting2()
fig_numb = 1


################## VMU2 Label File Name Processing ###################
print "################## VMU2 Label File Name Processing ##################"
train_dic = {}
train_dic2 = {}
test_dic = {}
test_dic2 = {}
values = []
values2 = []
for file in glob.glob(os.path.join(path, "*.bmp")):
    name_begin = file.rfind("\\")
    name_end = file.index("d")
    name = file[name_begin+1:name_end]

    label_numb = file.rfind("_")
    if label_numb != -1:
        label = file[label_numb+1]
    else:
        label = "original"

    file_name = file[name_begin+1:]
    dict2_end = file.rfind(".bmp")
    dict2 = file[name_begin+1:dict2_end]

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


################## VMU2 Training Data ################################
print "################## VMU2 Training Data ##################"
y_train = []
X_train = []
for train_image in train_images:
    # Read the image
    image =  sim.imread(train_image,True,None)  
    h = image.shape[0]
    w = image.shape[1]  
    label = train_dic[os.path.split(train_image)[1]]
    image = np.ravel(image)
    
    X_train.append(image)
    y_train.append(label)


################## VMU2 Test Data ####################################
print "################## VMU2 Test Data ##################"
y_test = []
X_test = []
test_images = cvutils.imlist(dst1)
for test_image in test_images:
    # Read the image
    image =  sim.imread(test_image,True,None)   
    label = test_dic[os.path.split(test_image)[1]]
    image = np.ravel(image)

    X_test.append(image)
    y_test.append(label)

################## VMU2 Plot CMC ######################################
print "################## VMU2 Plot CMC ##################"
rank,fpr,tpr,roc_auc = CMC_Data2(X_train,y_train,X_test,y_test,h,w,l2,l3,txt_dst,data_txt)