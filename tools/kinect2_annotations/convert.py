import cv2
import json
import uuid
import shutil
import sys
import os
sys.path.append(os.getcwd())
from sklearn.model_selection import train_test_split

# SELECTED_KEYPOINTS = [4, 5, 7, 8, 9, 11]


class Kinect:
    """
    Use data from kinect's camera to get new form json, which serve for training later.
    """
    def __init__(self, vid_link, json_link, selected_keypoints, dir, bbox=(0, 0, 1280, 720)):
        # selected_keypoints is an array of keypoints used
        # dir: path to save data

        self.vid_link = vid_link
        self.json_link = json_link
        self.selected_keypoints = selected_keypoints
        self.name = vid_link.split('/')[-1].split('.')[0]
        self.dir = dir
        self.bbox = bbox

    def split_video(self):
        img_dir = self.dir + '/images'
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
            cap = cv2.VideoCapture(self.vid_link)
            id = 0
            while True:
                _, img = cap.read()
                if img is None:
                    break
                name = self.dir + '/images/' + self.name + '_%04d' % (id) + '.jpg'
                img = cv2.resize(img, (1280, 720))
                cv2.imwrite(name, img)
                id += 1

    # create coco json
    def create_coco_json(self, save=True, num_img=None, before=True):
        # read data from kinect'camera and re-format it into json file

        with open(self.json_link) as f:
            file = json.load(f)
        json_file = {}
        images_arr = []
        annotations_arr = []

        categories = {}
        cat = []
        categories['id'] = '0'
        categories['name'] = 'person_0'
        categories['supercategory'] = 'person'
        kp = {}
        kp['0'] = 'spinebase'
        kp['1'] = 'spinemid'
        kp['2'] = 'neck'
        kp['3'] = 'head'
        kp['4'] = 'shoulderleft'
        kp['5'] = 'elbowleft'
        kp['6'] = 'wristleft'
        kp['7'] = 'handleft'
        kp['8'] = 'shoulderright'
        kp['9'] = 'elbowright'
        kp['10'] = 'wristright'
        kp['11'] = 'handright'
        kp['12'] = 'hipleft'
        kp['13'] = 'kneeleft'
        kp['14'] = 'ankleleft'
        kp['15'] = 'footleft'
        kp['16'] = 'hipright'
        kp['17'] = 'kneeright'
        kp['18'] = 'ankleright'
        kp['19'] = 'footright'
        kp['20'] = 'spineshoulder'
        kp['21'] = 'handtileft'
        kp['22'] = 'thumbleft'
        kp['23'] = 'handtiright'
        kp['24'] = 'thumbright'

        categories['keypoints'] = kp

        skeleton = dict()
        skeleton['0'] = [3, 2]
        skeleton['1'] = [2, 20]
        skeleton['2'] = [20, 1]
        skeleton['3'] = [1, 0]
        skeleton['4'] = [20, 8]
        skeleton['5'] = [8, 9]
        skeleton['6'] = [9, 10]
        skeleton['7'] = [10, 11]
        skeleton['8'] = [11, 23]
        skeleton['9'] = [11, 24]
        skeleton['10'] = [20, 4]
        skeleton['11'] = [4, 5]
        skeleton['12'] = [5, 6]
        skeleton['13'] = [6, 7]
        skeleton['14'] = [7, 21]
        skeleton['15'] = [7, 22]
        skeleton['16'] = [0, 12]
        skeleton['17'] = [0, 16]
        skeleton['18'] = [16, 17]
        skeleton['19'] = [17, 18]
        skeleton['20'] = [18, 19]
        skeleton['21'] = [12, 13]
        skeleton['22'] = [13, 14]
        skeleton['23'] = [14, 15]

        categories['skeletons'] = skeleton

        categories['selected_keypoint'] = self.selected_keypoints
        cat.append(categories)

        assert num_img is not None, 'You must split video first!'
        file = file[:num_img] if before else file[-num_img:]
        for id, f in enumerate(file):
            image = {}
            image['rights_holder'] = '---bestmonster---'
            image['license'] = '0'
            image['file_name'] = self.name + '_%04d' %id + '.jpg'
            image['url'] = self.dir + '/' + image['file_name']
            image['height'] = 1280
            image['width'] = 720
            image['id'] = id

            images_arr.append(image)
            for keypoints in f['bodies']:
                if keypoints['tracked']:
                    annotation = {}
                    annotation['image_id'] = id
                    annotation['iscrowd'] = 0
                    annotation['bbox'] = self.bbox
                    kp = []
                    num_keypoint = 25
                    for keypoint in keypoints['joints']:
                        confident = 2 if keypoint['jointType'] in self.selected_keypoints else 0
                        if keypoint['colorX'] is None:
                            num_keypoint -= 1
                            confident = 0
                            keypoint['colorX'] = 0
                            keypoint['colorY'] = 0
                        kp.append(round(keypoint['colorX'] * 1280) - self.bbox[0])
                        kp.append(round(keypoint['colorY'] * 720) - self.bbox[1])
                        kp.append(confident)
                    annotation['num_keypoint'] = num_keypoint
                    annotation['keypoints'] = kp
                    annotation['category_id'] = '0'
                    annotation['id'] = str(uuid.uuid1())
                    annotation['area'] = 1024

                    annotations_arr.append(annotation)

                    break      # 1 person was tracked in video

        license_arr = []
        lic_arr = dict()
        lic_arr['url'] = '---bestmonster---'
        lic_arr['id'] = '0'
        lic_arr['name'] = 'bestmonster'
        license_arr.append(lic_arr)

        json_file['images'] = images_arr
        json_file['licenses'] = license_arr
        json_file['annotations'] = annotations_arr
        json_file['categories'] = cat

        if save:
            with open(self.dir + '/annotations.json', 'w') as f:
                json.dump(json_file, f)
        return json_file

    # split vid and create json
    def process_kinect_data(self, before=True, draw_kp=False):
        self.split_video()
        num_img = len(os.listdir(self.dir + '/images'))
        data = self.create_coco_json(num_img=num_img, before=before)['annotations']

        # draw_kp: to check pose is correct or not
        if draw_kp:
            img_links = [self.dir + '/images/' + l for l in os.listdir(self.dir + '/images') ]
            img_links.sort()
            img_folder = self.dir + '/draw_imgs'
            if os.path.exists(img_folder):
                shutil.rmtree(img_folder)
            os.makedirs(img_folder)
            for id, img in enumerate(img_links):
                name = img.split('/')[-1]
                img = cv2.imread(img)
                kps = data[id]['keypoints']
                for sl_kp in self.selected_keypoints:
                    img = cv2.circle(img, center=(int(kps[3*sl_kp] + self.bbox[0]), int(kps[3*sl_kp + 1] + self.bbox[1])), color=(0, 0, 255), radius=5, thickness=3)
                    cv2.imwrite(img_folder + '/' + name, img)


def split(folder_path, save_link, test_size = 0.3):
    """
    input: folder_path: path to folder that contains images and annotations file
           save_link: path to save images and annotations file
           test_size: the ratio between test set and train set
    output: train set and test set, each set contains img's folder and annotations's file
    """
    # chia theo annos, luu thanh folder bao gom images, test json, train json

    image_arr = []
    annotations_arr = []
    img_id = 0
    for _, f in enumerate(os.listdir(folder_path)):
        if not os.path.isfile(folder_path + '/' + f + '/annotations.json'):
            continue
        with open(folder_path + '/' + f + '/annotations.json') as file:
            data = json.load(file)
        img = data['images']
        annos = data['annotations']
        for i in range(len(img)):
            img[i]['id'] = img_id
            annos[i]['image_id'] = img_id
            image_arr.append(img[i])
            annotations_arr.append(annos[i])
            img_id += 1

        # append images
        if not os.path.exists(save_link + '/se7en11/images'):
            os.makedirs(save_link + '/se7en11/images')

        img_link = folder_path + '/' + f + '/images'
        for l in os.listdir(img_link):
            shutil.copy(img_link + '/' + l, save_link + '/se7en11/images')

    annos_train, annos_test, img_train, img_test = train_test_split(annotations_arr, image_arr, test_size=test_size)

    # save train data
    train = dict()
    train['images'] = img_train
    train['licenses'] = data['licenses']
    train['annotations'] = annos_train
    train['categories'] = data['categories']

    with open(save_link + '/se7en11/train_annotations.json', 'w') as f:
        json.dump(train, f)

    # save test data
    test = dict()
    test['images'] = img_test
    test['annotations'] = annos_test
    test['categories'] = data['categories']
    test['licenses'] = data['licenses']

    with open(save_link + '/se7en11/test_annotations.json', 'w') as f:
        json.dump(test, f)


if __name__=='__main__':

    SELECTEDS_KPS = [3, 4, 8, 5, 9, 6, 10]

    """
    Extract images and data from kinect data
    """

    base_link = '/home/vietnguyen/LSTM_keypoint/database/Kinect v2 joints/v3'
    base_save_link = '/home/vietnguyen/LSTM_keypoint/test_kinect'

    # vid_folder: contains videos from kinectV2
    vid_folder = [base_link + '/vid/' + name for name in os.listdir(base_link + '/vid')]
    vid_folder.sort()

    # json_folder: contains json files corresponding to videos
    json_folder = [base_link + '/json/' + name for name in os.listdir(base_link + '/json')]
    json_folder.sort()

    for id in range(len(vid_folder)):
        # you must define the fix bbox for each video
        bbox = [150, 150, 500, 500]

        name = vid_folder[id].split('/')[-1].split('.')[0]
        save_link = base_save_link + '/' + name
        kinect = Kinect(vid_folder[id], json_folder[id], SELECTEDS_KPS, save_link, bbox)
        kinect.process_kinect_data()

    """
    Split data into train and test
    """

    folder_path = base_save_link
    split(folder_path, base_save_link, 0.2)