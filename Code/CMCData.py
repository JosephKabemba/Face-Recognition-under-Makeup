from setting import *

def CMC_Data(X_train,y_train,X_test,y_test,l2,l3,txt_dst,data_txt):
    n_components = 80
    pca = PCA(n_components=n_components, whiten=True).fit(X_train[:])
    eigenfaces = pca.components_.T.reshape((n_components, 304, 33))
    # project the input data on the eigenfaces orthonormal basis
    X_train_pca = pca.transform(X_train[:])
    X_test_pca = pca.transform(X_test[:])
    # Train a SVM classification model
    y_train = np.array(y_train)
    clf = SVC(decision_function_shape = 'ovr',probability=True)
    clf.fit(X_train_pca, y_train)

    # Quantitative evaluation of the model quality on the test set
    dec = clf.predict(X_test_pca)
    y_score = clf.predict_proba(X_test_pca)

    ################################################################
    # Directly using clf.predict_proba perserves higher accuancy.  #
    # Since on the menu, it shows that:                            #
    # The probability model is created using cross validation, so  #
    # the results can be slightly different than those obtained by #
    # predict. Also, it will produce meaningless results on very   #
    # small datasets.                                              #
    ################################################################
    count = 0
    for entry in range(len(dec)):
        if dec[entry] == y_test[entry]:
            count = count +1
    presentage = float(count)/float(len(dec))

    y_score = 1 - y_score
    yout = []
    y_score2 = y_score.tolist()
    for entry in y_score2:
        index = entry.index(max(entry))
        tmp = l2[index]
        yout.append(tmp)
    '''
    count = 0
    for entry in range(len(dec)):
        if yout[entry] == y_test[entry]:
            count = count +1
    presentage2 = float(count)/float(len(dec))
    '''
    rank = []
    y_score2 = y_score.tolist()
    for entry in y_score2:
        location = y_score2.index(entry)
        index = entry.index(max(entry))
        tmp = l2[index]
        if tmp == y_test[location]:
            rank.append(str(1.0))
        else:
            #Check the index of current name
            index2 = l3.index(y_test[location])
            #Check the number under this index, under this name
            probability = entry[index2]
            #Do the sort of the list entry
            #must deepcopy
            list_sorted = copy.deepcopy(entry)
            list.sort(list_sorted)
            #Find the rank of it
            rank_ind = list_sorted.index(probability)
            #Devide it into the total number 
            rank_result = float(rank_ind)/float(len(l2))
            rank.append(str(rank_result))

    f3 = open(txt_dst,"w")
    f3.write("PredictResu: "+str(dec) +"\n")
    f3.write("GroundTruth: "+str(y_test) +"\n")
    f3.write("Accuancy"+str(presentage) + " \n")
    f3.close()

    f4 = open(data_txt,"a")
    f4.write(str(presentage) + " \n")
    f4.close()

    # Compute ROC curve and ROC area for each class
    # Binarize the output
    y = label_binarize(y_test, l2)
    fpr, tpr, _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    return rank,fpr,tpr,roc_auc


def CMC_Data2(X_train,y_train,X_test,y_test,h,w,l2,l3,txt_dst,data_txt):
    n_components = 65
    pca = PCA(n_components=n_components, whiten=True).fit(X_train)
    eigenfaces = pca.components_.T.reshape((n_components, h, w))
    # project the input data on the eigenfaces orthonormal basis
    X_train_pca = pca.transform(X_train[:])
    X_test_pca = pca.transform(X_test[:])

    # Train a SVM classification model
    y_train = np.array(y_train)
    clf = SVC(decision_function_shape = 'ovr',probability=True)
    clf.fit(X_train_pca, y_train)

    # Quantitative evaluation of the model quality on the test set
    dec = clf.predict(X_test_pca)
    y_score = clf.predict_proba(X_test_pca)
    ################################################################
    # Directly using clf.predict_proba perserves higher accuancy.  #
    # Since on the menu, it shows that:                            #
    # The probability model is created using cross validation, so  #
    # the results can be slightly different than those obtained by #
    # predict. Also, it will produce meaningless results on very   #
    # small datasets.                                              #
    ################################################################
    count = 0
    for entry in range(len(dec)):
        if dec[entry] == y_test[entry]:
            count = count +1
    presentage = float(count)/float(len(dec))

    y_score = 1 - y_score
    yout = []
    y_score2 = y_score.tolist()
    for entry in y_score2:
        index = entry.index(max(entry))
        tmp = l2[index]
        yout.append(tmp)
    '''
    count = 0
    for entry in range(len(dec)):
        if yout[entry] == y_test[entry]:
            count = count +1
    presentage2 = float(count)/float(len(dec))
    '''
    rank = []
    y_score2 = y_score.tolist()
    for entry in y_score2:
        location = y_score2.index(entry)
        index = entry.index(max(entry))
        tmp = l2[index]
        if tmp == y_test[location]:
            rank.append(str(1.0))
        else:
            #Check the index of current name
            index2 = l3.index(y_test[location])
            #Check the number under this index, under this name
            probability = entry[index2]
            #Do the sort of the list entry
            #must deepcopy
            list_sorted = copy.deepcopy(entry)
            list.sort(list_sorted)
            #Find the rank of it
            rank_ind = list_sorted.index(probability)
            #Devide it into the total number 
            rank_result = float(rank_ind)/float(len(l2))
            rank.append(str(rank_result))

    f3 = open(txt_dst,"w")
    f3.write("PredictResu: "+str(dec) +"\n")
    f3.write("GroundTruth: "+str(y_test) +"\n")
    f3.write("Accuancy"+str(presentage) + " \n")
    f3.close()

    f4 = open(data_txt,"a")
    f4.write(str(presentage) + " \n")
    f4.close()

    # Compute ROC curve and ROC area for each class
    # Binarize the output
    y = label_binarize(y_test, l2)
    fpr, tpr, _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    return rank,fpr,tpr,roc_auc


def CMC_Data3(X_train,y_train,X_test,y_test,l2,l3,txt_dst,data_txt):

    # Train a SVM classification model
    y_train = np.array(y_train)
    clf = SVC(decision_function_shape = 'ovr',probability=True)
    clf.fit(X_train, y_train)

    # Quantitative evaluation of the model quality on the test set
    dec = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    ################################################################
    # Directly using clf.predict_proba perserves higher accuancy.  #
    # Since on the menu, it shows that:                            #
    # The probability model is created using cross validation, so  #
    # the results can be slightly different than those obtained by #
    # predict. Also, it will produce meaningless results on very   #
    # small datasets.                                              #
    ################################################################
    count = 0
    for entry in range(len(dec)):
        if dec[entry] == y_test[entry]:
            count = count +1
    presentage = float(count)/float(len(dec))

    y_score = 1 - y_score
    yout = []
    y_score2 = y_score.tolist()
    for entry in y_score2:
        index = entry.index(max(entry))
        tmp = l2[index]
        yout.append(tmp)
    '''
    count = 0
    for entry in range(len(dec)):
        if yout[entry] == y_test[entry]:
            count = count +1
    presentage2 = float(count)/float(len(dec))
    '''
    rank = []
    y_score2 = y_score.tolist()
    for entry in y_score2:
        location = y_score2.index(entry)
        index = entry.index(max(entry))
        tmp = l2[index]
        if tmp == y_test[location]:
            rank.append(str(1.0))
        else:
            #Check the index of current name
            index2 = l3.index(y_test[location])
            #Check the number under this index, under this name
            probability = entry[index2]
            #Do the sort of the list entry
            #must deepcopy
            list_sorted = copy.deepcopy(entry)
            list.sort(list_sorted)
            #Find the rank of it
            rank_ind = list_sorted.index(probability)
            #Devide it into the total number 
            rank_result = float(rank_ind)/float(len(l2))
            rank.append(str(rank_result))

    f3 = open(txt_dst,"w")
    f3.write("PredictResu: "+str(dec) +"\n")
    f3.write("GroundTruth: "+str(y_test) +"\n")
    f3.write("Accuancy"+str(presentage) + " \n")
    f3.close()

    f4 = open(data_txt,"a")
    f4.write(str(presentage) + " \n")
    f4.close()

    # Compute ROC curve and ROC area for each class
    # Binarize the output
    y = label_binarize(y_test, l2)
    fpr, tpr, _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    return rank,fpr,tpr,roc_auc
