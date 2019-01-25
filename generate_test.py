import numpy as np
import pandas as pd
from pathlib import Path
from load_data import *
from utils import *
import subprocess
from PIL import Image
import random
import math


def random_test(person_id, date):

    minute_based_lifelog, visual_concept_df = load_data_as_hdf(person_id) 
    # Create test folder for person_id
    test_path = Path.cwd().parent / 'data' / 'test' / person_id / date
    train_path = Path.cwd().parent / 'data' / 'train' / person_id / date
    ground_truth_path = Path.cwd().parent / 'data' / 'ground_truth' / person_id /date
    create_folder(str(test_path))
    create_folder(str(train_path))
    create_folder(str(ground_truth_path))

    @time_this
    def gen_random_test():

        # Shuffle data to create test for ImageCLEF (test minutes with images only)
        # The percentage of test data is 25%
        lifelog_date_df = minute_based_lifelog[minute_based_lifelog['minute_ID'].str.match(\
            '{}_{}'.format(person_id, date.replace('_', '')))]
        lifelog_date_df = lifelog_date_df.query('img00_id == img00_id')

        ground_truth_df = lifelog_date_df.sample(frac=0.25).sort_index()
        ground_truth_df.to_hdf(str(ground_truth_path / 'minute_based_table.hdf'), 'data', mode='w')
        ground_truth_df.to_csv(str(ground_truth_path / 'minute_based_table.csv'), index=False, na_rep='NULL')

        train_df = lifelog_date_df[lifelog_date_df.index.isin(ground_truth_df.index.tolist())]
        train_df.to_hdf(str(train_path / 'minute_based_table.hdf'), 'data', mode='w')
        train_df.to_csv(str(train_path / 'minute_based_table.csv'), index=False, na_rep='NULL')

        return train_df, ground_truth_df


    @time_this
    def gen_concepts(train_df, ground_truth_df):
        # Retrieve image paths and create test images
        ground_truth_vc_df = get_image_concepts(ground_truth_df, visual_concept_df)
        ground_truth_vc_df.to_hdf(str(ground_truth_path / '{}_categories_attr_concepts.hdf'.format(person_id)), 'data', mode='w')
        ground_truth_vc_df.to_csv(str(ground_truth_path / '{}_categories_attr_concepts.csv'.format(person_id)), index=False, na_rep='NULL')

        hash_image_ids = []
        image_ids = []
        image_paths = []
        for v in ground_truth_vc_df.values:
            image_id, image_path, *r = v
            image_ids.append(image_id)
            image_paths.append(image_path)
            hash_image_ids.append("%032x.JPG" % random.getrandbits(128))
        map_df = pd.DataFrame({'image_id' : image_ids, 'image_path': image_paths, 'hash_id': hash_image_ids},\
                columns=['image_id', 'image_path', 'hash_id'])
        map_df.to_hdf(str(ground_truth_path / 'map_tables.hdf'), 'data', mode='w')
        map_df.to_csv(str(ground_truth_path / 'map_tables.csv'), index=False, na_rep='NULL')

        test_vc_df = ground_truth_vc_df.drop(['image_id', 'image_path'], axis=1)
        test_vc_df.insert(0, 'image_name', hash_image_ids)
        test_vc_df.to_hdf(str(test_path / '{}_categories_attr_concepts.hdf'.format(person_id)), 'data', mode='w')
        test_vc_df.to_csv(str(test_path / '{}_categories_attr_concepts.csv'.format(person_id)), index=False, na_rep='NULL')

        train_vc_df = get_image_concepts(train_df, visual_concept_df)
        train_vc_df.to_hdf(str(train_path / '{}_categories_attr_concepts.hdf'.format(person_id)), 'data', mode='w')
        train_vc_df.to_csv(str(train_path / '{}_categories_attr_concepts.csv'.format(person_id)), index=False, na_rep='NULL')

        return train_vc_df, map_df


    @time_this
    def gen_tests(map_df, ground_truth_df):
        test_models = parse_minute_based_lifelog(ground_truth_df)
        test_df = ground_truth_df.drop(['minute_ID', 'utc_time', 'local_time'], axis=1)
        wear_imgs = []
        hash_ids = []
        for v in test_models.values:
            _, model = v
            img_ids = [img_id for img_id in model.wear_imgs if isinstance(img_id, str)]
            hids = [map_df.query('image_id == "{}"'.format(img_id)).iloc[0]['hash_id'] for img_id in img_ids] 
            wear_imgs += img_ids
            hash_ids += hids
        for v in list(set(zip(wear_imgs, hash_ids))):
            img_id, hash_id = v
            test_df.replace(img_id, hash_id, inplace=True)
        test_df.to_hdf(str(test_path / 'minute_based_table.hdf'), 'data', mode='w') 
        test_df.to_csv(str(test_path / 'minute_based_table.csv'), index=False, na_rep='NULL')

        # Create test images and delete all time information
        src_path = Path.cwd().parent / 'data' / person_id / 'Autographer'
        dest_path = Path.cwd().parent / 'data' / 'test' / person_id / date / 'Autographer'
        create_folder(str(dest_path), exist_ok=False)
        for v in map_df.values:
            _, image_path, hash_image_id = v
            src_image_path = str(src_path / image_path)
            dest_image_path = str(dest_path / hash_image_id)
            image = Image.open(src_image_path)
            image_without_exif = Image.new(image.mode, image.size)
            image_without_exif.putdata(list(image.getdata()))
            image_without_exif.save(dest_image_path)


    @time_this
    def gen_train_images(train_vc_df):
        # Create train images
        src_path = Path.cwd().parent / 'data' / person_id / 'Autographer'
        dest_path = Path.cwd().parent / 'data' / 'train' / person_id / date / 'Autographer'
        create_folder(str(dest_path), exist_ok=False)
        for v in train_vc_df.values:
            image_path = v[1]
            src_image_path = str(src_path / image_path)
            cmd = 'cp {} {}'.format(str(src_image_path), str(dest_path))
            subprocess.call(cmd, shell=True)


    @time_this
    def get_image_concepts(lifelog_df, visual_concept_df):
        lifelog_models = parse_minute_based_lifelog(lifelog_df)
        df_concepts = []
        for v in lifelog_models.values:
            _, model = v
            temp_df = [visual_concept_df.query('image_id == "{}"'.format(img_id)) for img_id in model.wear_imgs]
            df_concepts.append(pd.concat(temp_df))
        df_concepts = pd.concat(df_concepts)
        return df_concepts

    def main():
        train_df, ground_truth_df = gen_random_test()
        train_vc_df, map_df = gen_concepts(train_df, ground_truth_df)
        gen_tests(map_df, ground_truth_df)
        gen_train_images(train_vc_df)
        
    main()

if __name__ == '__main__':
    random_test('u1', '2018_05_03')
