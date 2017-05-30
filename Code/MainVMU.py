from setting import *
import VMU
import VMU2PCA
import VMU3HOG

def PlotCMC():
    figure(1)
    this1 = VMU.rank
    pca2 = VMU2PCA.rank
    hog3 = VMU3HOG.rank
    pdf_dst = "FinalCMC"

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    tpr = [this1,pca2,hog3]
    name = ['DTCWT+LBP','PCA','HOG']
    length = len(tpr[0])
    for i in range(len(tpr)):
        list.sort(tpr[i])

    for i, color in zip(range(len(tpr)), colors):
        plt.plot(range(length), tpr[i], color=color,label = name[i], markersize=0.5)
    plt.ylim(0,1.1)
    #plt.ylim(0.6,1.01)
    plt.xlim(-5,151)
    plt.legend(loc="lower right")
    title('CMC Curve', color='#000000')
    plt.savefig(pdf_dst)
    #plt.show()


def PlotROC():
    figure(2)
    pdf_dst = "FinalROC"
    lw = 1

    this1_fpr = VMU.fpr
    this1_tpr = VMU.tpr
    this1_auc = VMU.roc_auc

    pca2_fpr = VMU2PCA.fpr
    pca2_tpr = VMU2PCA.tpr
    pca2_auc = VMU2PCA.roc_auc

    hog3_fpr = VMU3HOG.fpr
    hog3_tpr = VMU3HOG.tpr
    hog3_auc = VMU3HOG.roc_auc

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','antiquewhite','purple'])
    fpr = [this1_fpr,pca2_fpr,hog3_fpr]
    tpr = [this1_tpr,pca2_tpr,hog3_tpr]
    roc_auc = [this1_auc,pca2_auc,hog3_auc]

    name = ['DTCWT+LBP','PCA','HOG']



    for i, color in zip(range(len(tpr)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1:0.4f})'''.format(name[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    plt.savefig(pdf_dst)
    #plt.show()

PlotROC()
PlotCMC()
