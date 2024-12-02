import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
d = {}

d["road"] = {'N1':[], 'N2':['drivable fallback'], 'N3':[]}

d["drivable fallback"] = {'N1':[], 'N2':["road"], 'N3':[]}

d["sidewalk"] = {'N1':[], 'N2':["non drivable fallback"], 'N3':[]}

d["non drivable fallback"] = {'N1':[], 'N2':["sidewalk"], 'N3':[]}

d["person"] = {'N1':[], 'N2':["rider"], 'N3':[]}

d["rider"] = {'N1':[], 'N2':["person"], 'N3':[]}

d["motorcycle"] = {'N1':["bicycle"], 'N2':["autorickshaw", "car", "truck",

                            "bus", "vehicle fallback"], 'N3':[]}

d["bicycle"] = {'N1':["motorcycle"], 'N2':["autorickshaw", "car", "truck",

                            "bus", "vehicle fallback"], 'N3':[]}

d["autorickshaw"] = {'N1':["car"], 'N2':["bicycle", "motorcycle", "truck",

                            "bus", "vehicle fallback"], 'N3':[]}

d["car"] = {'N1':["autorickshaw"], 'N2':["bicycle", "motorcycle", "truck",

                            "bus", "vehicle fallback"], 'N3':[]}

d["truck"] = {'N1':["bus", "vehicle fallback"], 'N2':["motorcycle",

                            "bicycle", "autorickshaw", "car"], 'N3':[]}

d["bus"] = {'N1':["truck", "vehicle fallback"], 'N2':["motorcycle",

                            "bicycle", "autorickshaw", "car"], 'N3':[]}

d["vehicle fallback"] = {'N1':["truck", "bus"], 'N2':["motorcycle",

                            "bicycle", "autorickshaw", "car"], 'N3':[]}

d["curb"] = {'N1':["wall"], 'N2':["fence", "guard rail", "billboard",

                            "traffic sign", "traffic light", "pole", "obs-str-bar-fallback"], 'N3':[]}

d["wall"] = {'N1':["curb"], 'N2':["fence", "guard rail", "billboard",

                            "traffic sign", "traffic light", "pole", "obs-str-bar-fallback"], 'N3':[]}

d["fence"] = {'N1':["guard rail"], 'N2':["curb", "wall", "billboard",

                            "traffic sign", "traffic light", "pole", "obs-str-bar-fallback"], 'N3':[]}

d["guard rail"] = {'N1':["fence"], 'N2':["curb", "wall", "billboard",

                            "traffic sign", "traffic light", "pole", "obs-str-bar-fallback"], 'N3':[]}

d["billboard"] = {'N1':["traffic sign", "traffic light"], 'N2':["curb",

                            "wall", "fence", "guard rail", "pole", "obs-str-bar-fallback"], 'N3':[]}

d["traffic sign"] = {'N1':["billboard", "traffic light"], 'N2':["curb",

                            "wall", "fence", "guard rail", "pole", "obs-str-bar-fallback"], 'N3':[]}

d["traffic light"] = {'N1':["billboard", "traffic sign"], 'N2':["curb",

                            "wall", "fence", "guard rail", "pole", "obs-str-bar-fallback"], 'N3':[]}

d["pole"] = {'N1':["obs-str-bar-fallback"], 'N2':["curb", "wall", "fence",

                             "guard rail", "billboard", "traffic light", "traffic sign"], 'N3':[]}

d["obs-str-bar-fallback"] = {'N1':["pole"], 'N2':["curb", "wall", "fence",

                             "guard rail",  "billboard", "traffic light", "traffic sign"], 'N3':[]}

d["building"] = {'N1':["bridge"], 'N2':["vegetation"], 'N3':[]}

d["bridge"] = {'N1':["building"], 'N2':["vegetation"], 'N3':[]}

d["vegetation"] = {'N1':[], 'N2':["bridge", "building"], 'N3':[]}

d["sky"] = {'N1':[], 'N2':[], 'N3':[]}


CLASSES = ['road', 'drivable fallback', 'sidewalk', 'non drivable fallback',

           'person', 'rider', 'motorcycle', 'bicycle', 'autorickshaw',

           'car', 'truck', 'bus', 'vehicle fallback', 'curb', 'wall',

           'fence', 'guard rail', 'billboard', 'traffic sign',

           'traffic light', 'pole', 'obs-str-bar-fallback', 'building',

           'bridge', 'vegetation', 'sky']

 

class2id = {}

for i in range(26):
    class2id[CLASSES[i]] = i 


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist,num_classes=26):
    #return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  
    r=[]
    r1=[]
    r2=[]
    r3=[]
    r4=[]
    den = hist.sum(1) + hist.sum(0) - np.diag(hist)
    
    for i in range(num_classes):
        
        D=den[i]
        N=hist[i][i]
        T0=N 

        NT1=sum(hist[i]) - hist[i][i] # sum of current class pixesl predicted wrongly NT1 is last negative term
        NT2 = 0 #NG2=sum(hist[i]) - hist[i][i] # sum of current class pixesl predicted wrongly
        T1=0
        T2=0
        T3=0
        if [class2id[x] for x in d[CLASSES[i]]['N1']]!=[]:
            for ele in [class2id[x] for x in d[CLASSES[i]]['N1']]:
                T1+=hist[i][ele]
        if [class2id[x] for x in d[CLASSES[i]]['N2']]!=[]:
            for ele in [class2id[x] for x in d[CLASSES[i]]['N2']]:
                T2+=hist[i][ele]

        T3 = sum(hist[i]) - T0 - T1 - T2
        Td = [1, 2, 3] #treedistance/2
        
        W = [0.5 , -1 , -1.5] # td/2 for 1st term  and -td/2 for other terms 

        #N1 = T0+0.75*T1 # iou + iou of sibling(1st nearest)
        #N2 = T0+(0.75*T1)+(0.5*T2) # iou + weighted iou1 + weighted iou2 of sibling(1st nearest) and next level (2nd nearest)
        #N3 = T0+(0.75*T1)+(0.5*T2) -1*T3  # N2 with weighted negative penality for left over classes
        #N4 = T0+(0.75*T1) - (0.5*T2) - (1*T3)  # N1 with weighted negative penality for left over classes
        N1 = T0+W[0]*T1 # iou + iou of sibling(1st nearest)
        N4 = T0+(W[0]*T1) +(W[1]*T2) + (W[2]*T3)  # N1 with weighted negative penality for left over classes

        r.append(N/D)
        r1.append(N1/D)
        #r2.append(N2/D)
        #r3.append(N3/D)
        r4.append(N4/D)
    return r,r1,r4 #r,r1,r2,r3,r4


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):
    """
    Compute IoU given the predicted colorized images and 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
      info = json.load(fp)
    num_classes = int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=str)
    mapping = np.array(info['label2train'], dtype=int)
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'val.txt')
    label_path_list = join(devkit_dir, 'label.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    # print(pred_imgs[0])
    # #exit()
    pred_imgs = ['_'.join(s[24:].split('/')) for s in pred_imgs] #10,16
    # pred_imgs = [s.split('/')[3]+"_"+s.split('/') for s in pred_imgs]
    # print(pred_imgs[0])
    # exit()
    pred_imgs = [pred_dir+'/'+ x for x in pred_imgs]
    # print(pred_imgs[0],gt_imgs[0])
    # print(pred_imgs[:20])
    # print(gt_imgs[:20])
    for ind in range(len(gt_imgs)):
        pred=np.array(Image.open(pred_imgs[ind]))
        #pred = np.array(pr.resize((1920,1080)) )
        label=np.array(Image.open(gt_imgs[ind]))
        #label = np.array(la.resize((1920,1080)))
        # print(pred)
        # print(' == ')
        # print(label)
        # exit()

        #print(f'sahpe of pred : {pred.shape}, label : {label.shape}')

        #label = label_mapping(label, mapping)
        #print(f'labe :{label}')
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        histpm = fast_hist(label.flatten(), pred.flatten(), num_classes)
        # hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        #print(f'shape of hist :{hist.shape}')
        #print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(histpm,num_classes)[0])))
        if ind > 0 and ind % 50 == 0:
           #print(f'den :{hist.sum(1) + hist.sum(0) - np.diag(hist)}')
            print(' {:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist,num_classes)[0])))


    #mIoUs, mIoUs1, mIoUs2,mIoUs3,mIoUs4 = per_class_iu(hist,num_classes)
    mIoUs, mIoUs1,mIoUs4 = per_class_iu(hist,num_classes)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)) + ':\t' + str(round(mIoUs1[ind_class] * 100, 2)) + ':\t' + str(round(mIoUs4[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2))+ ':\t' + str(round(np.nanmean(mIoUs1) * 100, 2)) + ':\t' + str(round(np.nanmean(mIoUs4) * 100, 2)))
    return  mIoUs,  mIoUs1, mIoUs4  # mIoUs,  mIoUs1, mIoUs2, mIoUs3, mIoUs4

    '''for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)) + ':\t' + str(round(mIoUs1[ind_class] * 100, 2)) + ':\t' + str(round(mIoUs2[ind_class] * 100, 2))
               + ':\t' + str(round(mIoUs3[ind_class] * 100, 2)) + ':\t' + str(round(mIoUs4[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2))+ ':\t' + str(round(np.nanmean(mIoUs1) * 100, 2))+ ':\t' + str(round(np.nanmean(mIoUs2) * 100, 2))
          + ':\t' + str(round(np.nanmean(mIoUs3) * 100, 2))+ ':\t' + str(round(np.nanmean(mIoUs4) * 100, 2)))
    return mIoUs,  mIoUs1, mIoUs2, mIoUs3, mIoUs4'''


def main(args):
   compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', default='/ssd_scratch/cvit/furqan.shaik/',type=str, help='directory which stores CityScapes val gt images')
    parser.add_argument('--pred_dir', default='predictions', type=str, help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default='iddaw', help='base directory of cityscapes')
    args = parser.parse_args()
    main(args)

## python 
