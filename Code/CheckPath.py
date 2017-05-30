from setting import *

def CheckPathSetting1():
    path = ".\\before\W\VMU" #origional data
    path2 = ".\\before\W\VMU\VMU_final"
    if os.path.exists(path2):
        shutil.rmtree(path2)
        os.makedirs(path2)
    else:
        os.makedirs(path2)

    txt_train_dst = ".\\before\W\VMU\VMU_final\class_train.txt"
    txt_test_dst = ".\\before\W\VMU\VMU_final\class_test.txt"
    dst1 = r".\\before\W\VMU\VMU_final\Testing"
    dst2 = r".\\before\W\VMU\VMU_final\Training"

    if not os.path.exists(dst1):
        os.makedirs(dst1)
    if not os.path.exists(dst2):
        os.makedirs(dst2)

    return path, path2, txt_train_dst, txt_test_dst, dst1, dst2


def CheckPathSetting2():
    path = ".\\before\W\VMU2" #origional data
    path2 = ".\\before\W\VMU2\VMU_final"
    if os.path.exists(path2):
        shutil.rmtree(path2)
        os.makedirs(path2)
    else:
        os.makedirs(path2)

    txt_train_dst = ".\\before\W\VMU2\VMU_final\class_train.txt"
    txt_test_dst = ".\\before\W\VMU2\VMU_final\class_test.txt"
    dst1 = r".\\before\W\VMU2\VMU_final\Testing"
    dst2 = r".\\before\W\VMU2\VMU_final\Training"

    if not os.path.exists(dst1):
        os.makedirs(dst1)
    if not os.path.exists(dst2):
        os.makedirs(dst2)
    
    return path, path2, txt_train_dst, txt_test_dst, dst1, dst2

def CheckPathSetting3():
    path = ".\\before\W\YMU" #origional data
    path2 = ".\\before\W\YMU\YMU_final"
    if os.path.exists(path2):
        shutil.rmtree(path2)
        os.makedirs(path2)
    else:
        os.makedirs(path2)

    txt_train_dst = ".\\before\W\YMU\YMU_final\class_train.txt"
    txt_test_dst = ".\\before\W\YMU\YMU_final\class_test.txt"
    dst1 = r".\\before\W\YMU\YMU_final\Testing"
    dst2 = r".\\before\W\YMU\YMU_final\Training"

    if not os.path.exists(dst1):
        os.makedirs(dst1)
    if not os.path.exists(dst2):
        os.makedirs(dst2)

    return path, path2, txt_train_dst, txt_test_dst, dst1, dst2

def CheckPathSetting4():
    path = ".\\before\W\YMU2" #origional data
    path2 = ".\\before\W\YMU2\YMU_final"
    if os.path.exists(path2):
        shutil.rmtree(path2)
        os.makedirs(path2)
    else:
        os.makedirs(path2)

    txt_train_dst = ".\\before\W\YMU2\YMU_final\class_train.txt"
    txt_test_dst = ".\\before\W\YMU2\YMU_final\class_test.txt"
    dst1 = r".\\before\W\YMU2\YMU_final\Testing"
    dst2 = r".\\before\W\YMU2\YMU_final\Training"

    if not os.path.exists(dst1):
        os.makedirs(dst1)
    if not os.path.exists(dst2):
        os.makedirs(dst2)

    return path, path2, txt_train_dst, txt_test_dst, dst1, dst2