import numpy as np
import pandas as pd
from pathlib import Path
from load_data import *
from utils import *
import subprocess
from PIL import Image
import random
import math


def get_random_list(num_rands=2003):
    rand_list = ['%032x' % random.getrandbits(128) for i in range(num_rands)]
    return rand_list


def get_image_concepts(lifelog_df, visual_concept_df):
    lifelog_models = parse_minute_based_lifelog(lifelog_df)
    df_concepts = []
    for v in lifelog_models.values:
        _, model = v
        temp_df = [visual_concept_df.query('image_id == "{}"'.format(img_id)) for img_id in model.wear_imgs if isinstance(img_id, str)]
        temp_cam_df = [visual_concept_df.query('image_id == "{}"'.format(img_id)) for img_id in model.cam_imgs if isinstance(img_id, str)]
        if len(temp_df) > 0:
            df_concepts.append(pd.concat(temp_df))
        if len(temp_cam_df) > 0:
            df_concepts.append(pd.concat(temp_cam_df))
    if len(df_concepts) > 0:
        df_concepts = pd.concat(df_concepts)
    return df_concepts


def gen_random_test(person_id, date):

    # Load minute based lifelog data, visual concepts and filter data
    autographer_path = Path.cwd().parent / 'data' / person_id / 'Autographer'
    personal_cam_path = Path.cwd().parent / 'data' / person_id
    test_path = Path.cwd().parent / 'data' / person_id / 'test' / date
    ground_truth_path = Path.cwd().parent / 'data' / person_id / 'ground_truth' / date
    create_folder(str(test_path), exist_ok=False)
    create_folder(str(ground_truth_path), exist_ok=False)

    minute_based_lifelog, visual_concept_df = load_data_as_hdf(person_id)
    lifelog_date_df = minute_based_lifelog[minute_based_lifelog['minute_ID'].str.match(\
        '{}_{}'.format(person_id, date.replace('_', '')))]
    cam_images = [image_name for image_name in os.listdir(str(personal_cam_path / '{}_photos'.format(person_id))) if image_name.startswith(date.replace('_', '-'))]
    num_rands = len(lifelog_date_df.index) + len(os.listdir(str(autographer_path / date))) + len(cam_images)
    rand_list = get_random_list(num_rands=num_rands)
    cur = 0

    # Shuffle and generate random hash id for minute based lifelog
        # Ground-truth part 
    minute_shuffle_df = lifelog_date_df.sample(frac=1)
    minutes_map_df = minute_shuffle_df.iloc[:,0:3]
    num_rows = len(minutes_map_df)
    hash_min_ids = rand_list[cur:cur+num_rows]
    cur += num_rows
    minutes_map_df.insert(1, 'minute_hash_id', hash_min_ids)
        # Test part
    minute_shuffle_df.drop(['minute_ID', 'utc_time', 'local_time'], axis=1, inplace=True)
    minute_shuffle_df.insert(0, 'minute_id', hash_min_ids) 

    # Shuffe and generate random hash id and get visual concepts for images
        # Ground-truth part
    image_filter_df = lifelog_date_df.query('img00_id == img00_id | cam00_id == cam00_id') 
    vc_df = get_image_concepts(image_filter_df, visual_concept_df)
    image_map_df = vc_df.iloc[:,0:2]
    num_rows = len(image_map_df.index)
    hash_min_ids = ['{}.JPG'.format(hash_id) for hash_id in rand_list[cur:cur+num_rows]]
    cur += num_rows
    image_map_df.insert(0, 'image_hash_id', hash_min_ids)
        # Test part
    vc_df.drop(['image_id', 'image_path'], axis=1, inplace=True)
    vc_df.insert(0, 'image_id', hash_min_ids)

    # Encrypt minute based lifelog image_id
    for v in image_map_df.values:
        hash_id, image_id, _ = v
        minute_shuffle_df.replace(image_id, hash_id, inplace=True)

    # Generate test images and clear information
    dest_cam = test_path / '{}_photos'.format(person_id)
    dest_autograph = test_path / 'Autographs'
    create_folder(str(dest_cam), exist_ok=False)
    create_folder(str(dest_autograph), exist_ok=False)
    for v in image_map_df.values:
        hash_image_id, _, image_path = v
        if image_path.split('/')[0] == '{}_photos'.format(person_id):
            src_image_path = str(personal_cam_path / image_path)
            dest_image_path = str(dest_cam / hash_image_id)
        else:
            src_image_path = str(autographer_path / image_path)
            dest_image_path = str(dest_autograph / hash_image_id)
        image = Image.open(src_image_path)
        image_without_exif = Image.new(image.mode, image.size)
        image_without_exif.putdata(list(image.getdata()))
        image_without_exif.save(dest_image_path)

    # Save the whole results as new test and ground truth
    minute_shuffle_df.to_csv(str(test_path / 'minute_based_table.csv'), index=False, na_rep='NULL')
    minutes_map_df.to_csv(str(ground_truth_path / '{}_{}_minute_map_table.csv'.format(person_id, date)))
    lifelog_date_df.to_csv(str(ground_truth_path / '{}_{}_minute_based_table.csv'.format(person_id, date)), index=False, na_rep='NULL')

    vc_df.to_csv(str(test_path / '{}_{}_categories_attr_concepts'.format(person_id, date)), index=False, na_rep='NULL')
    image_map_df.to_csv(str(ground_truth_path / '{}_{}_image_map_table.csv'.format(person_id, date)), index=False)


if __name__ == '__main__':
    gen_random_test('u1', '2018_05_03')
