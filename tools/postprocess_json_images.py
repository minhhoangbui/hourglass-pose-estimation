import json
import os
import glob

#######
# Filter images
#######
# root = '/home/hoang/datasets/JR-Wrist/test/*/'
# subdirs = glob.glob(root)
# for subdir in subdirs:
#     print(subdir)
#     data = dict()
#     data['images'] = []
#
#     file_path = os.path.join(subdir, 'annotations.json')
#     image_path = os.path.join(subdir, 'images')
#
#     with open(file_path, 'r') as fp:
#         json_file = json.load(fp)
#
#     used_ids = []
#     unused_imgs = []
#
#     for anno in json_file['annotations']:
#         if anno['image_id'] not in used_ids:
#             used_ids.append(anno['image_id'])
#
#     for image in json_file['images']:
#         if image['id'] not in used_ids:
#             unused_imgs.append(image['file_name'])
#         else:
#             data['images'].append(image)
#     data['licenses'] = json_file['licenses']
#     data['annotations'] = json_file['annotations']
#     data['categories'] = json_file['categories']
#
#     with open(os.path.join(subdir, 'annotations.json'), 'w') as fp:
#         json.dump(data, fp)
#
#     for fname in unused_imgs:
#         full_path = os.path.join(image_path, fname)
#         if os.path.isfile(full_path):
#             os.remove(os.path.join(image_path, fname))

######
# Replace image_id
######
root = '/home/hoang/datasets/JR-Wrist/test/*/'
subdirs = glob.glob(root)
for subdir in subdirs:
    print(subdir)
    file_path = os.path.join(subdir, 'annotations.json')
    with open(file_path, 'r') as fp:
        json_file = json.load(fp)

    for anno in json_file['annotations']:
        anno['image_id'] = int(anno['image_id'].split('_')[1])

    for image in json_file['images']:
        image['id'] = int(image['id'].split('_')[1])

    with open(os.path.join(subdir, "annotations.json"), 'w') as fp:
        json.dump(json_file, fp)

#####
#####
# data = dict()
# data['images'] = []
# data['licenses'] = []
# data['annotations'] = []
# data['categories'] = []
#
# root = '/home/hoang/datasets/JR-Wrist/AVC*/'
# subdirs = glob.glob(root)
# for subdir in subdirs:
#     file_path = os.path.join(subdir, 'processed_annotations.json')
#     with open(file_path, 'r') as fp:
#         json_file = json.load(fp)
#
#     current_nbr_images = len(data['images'])
#     for image in json_file['images']:
#         image['id'] += current_nbr_images
#         fulldir, fname = os.path.split(image['url'])
#         url, _ = os.path.split(fulldir)
#         image['url'] = os.path.join(url, fname)
#     for anno in json_file['annotations']:
#         anno['image_id'] += current_nbr_images
#     data['images'].extend(json_file['images'])
#     data['annotations'].extend(json_file['annotations'])
#
#
# data['licenses'] = json_file['licenses']
# data['categories'] = json_file['categories']
#
# with open(os.path.join('/home/hoang/datasets/JR-Wrist/', 'processed_annotations.json'), 'w') as fp:
#     json.dump(data, fp)

#####
#####
# data = dict()
# data['images'] = []
# data['licenses'] = []
# data['annotations'] = []
# data['categories'] = []
#
# image_dir = '/home/hoang/datasets/JR-Wrist/images/'
# json_path = '/home/hoang/datasets/JR-Wrist/_annotations.json'
# images = []
# for file in glob.glob(image_dir + '*.jpg'):
#     images.append(os.path.split(file)[1])
#
# with open(json_path, 'r') as fp:
#     json_file = json.load(fp)
#
# print(len(json_file['images']))
# for image in json_file['images']:
#     if image['file_name'] in images:
#         data['images'].append(image)
# print(len(data['images']))
# data['annotations'] = json_file['annotations']
# data['licenses'] = json_file['licenses']
# data['categories'] = json_file['categories']
#
# with open(os.path.join('/home/hoang/datasets/JR-Wrist/', '_annotations.json'), 'w') as fp:
#     json.dump(data, fp)

#####
#####

# file_path = '/home/hoang/datasets/JR-Wrist/annotations.json'
# with open(file_path, 'r') as fp:
#     json_file = json.load(fp)
# for anno in json_file['annotations']:
#     anno['image_id'] = int(anno['image_id'])
# for image in json_file['images']:
#     image['id'] = int(image['id'])
# with open(os.path.join('/home/hoang/datasets/JR-Wrist/', '_annotations.json'), 'w') as fp:
#     json.dump(json_file, fp)



