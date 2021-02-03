from pycocotools.coco import COCO
import json


def extract_hand_joints(org_annotation_file, dest_annotation_file, selected_ids):
    assert isinstance(selected_ids, list)
    db = dict()

    original_coco = COCO(org_annotation_file)
    org_image_ids = original_coco.getImgIds()

    final_anno_ids = []
    final_images_ids = []
    for img_id in org_image_ids:
        anno_ids = original_coco.getAnnIds(imgIds=img_id)
        is_removed = True

        for anno_id in anno_ids:
            anno = original_coco.loadAnns(anno_id)[0]

            if not anno['lefthand_valid'] and not anno['righthand_valid']:
                continue
            is_removed = False
            final_anno_ids.append(anno_id)
        if not is_removed:
            final_images_ids.append(img_id)

    db['images'] = list()
    for img_id in final_images_ids:
        db['images'].append(original_coco.loadImgs(img_id)[0])

    db['annotations'] = list()
    for anno_id in final_anno_ids:
        anno = original_coco.loadAnns(anno_id)[0]
        l_kps = anno['lefthand_kpts']

        kps = []
        for idx in selected_ids:
            kps.append(l_kps[3 * idx])
            kps.append(l_kps[3 * idx + 1])
            kps.append(l_kps[3 * idx + 2])

        r_kps = anno['righthand_kpts']
        for idx in selected_ids:
            kps.append(r_kps[3 * idx])
            kps.append(r_kps[3 * idx + 1])
            kps.append(r_kps[3 * idx + 2])
        anno['keypoints'] = kps
        anno.pop('righthand_valid')
        anno.pop('lefthand_valid')
        anno.pop('face_valid')
        anno.pop('foot_valid')
        anno.pop('righthand_kpts')
        anno.pop('lefthand_kpts')
        anno.pop('face_kpts')
        anno.pop('foot_kpts')
        anno.pop('righthand_box')
        anno.pop('lefthand_box')
        anno.pop('face_box')

        db['annotations'].append(anno)
    with open(dest_annotation_file, 'w') as fp:
        json.dump(db, fp)


if __name__ == '__main__':
    o_train = '/home/vietnguyen/datasets/coco/coco_wholebody_train_v1.0.json'
    o_val = '/home/vietnguyen/datasets/coco/coco_wholebody_val_v1.0.json'
    d_train = '/home/hoangbm/hourglass-pose-estimation/data/hands/hands_train.json'
    d_val = '/home/hoangbm/hourglass-pose-estimation/data/hands/hands_val.json'
    selected_ids = [0, 2, 4, 5, 8, 9, 12, 13, 16, 17, 20]
    extract_hand_joints(o_train, d_train, selected_ids)
    extract_hand_joints(o_val, d_val, selected_ids)