from utils import *
import os.path as osp
from pathlib import Path
from models import *
import numpy as np
import pandas as pd


def is_valid(data):
    return data != 'NULL' and data != 'nan'


@time_this
def load_csv_file(folder, file_name):
    file_path = folder / file_name
    df = pd.read_csv(str(file_path), engine='python')
    return df


def parse_minute_based_lifelog(df):
    minute_ids = []
    lifelog_models = []
    for v in df.values:
        minute_id, utc_time, local_time, time_zone, lat, lng, name,\
        song, activity, steps, calories, historic_glucose, scan_glucose,\
        heart_rate, distance, *r = v
        wear_imgs_id = r[:20]
        cam_imgs_id = r[20:]

        time = Time(utc_time, local_time, time_zone)
        geolocation = Geolocation(lat, lng, name)
        activity_model = Activity(activity, steps, calories, historic_glucose, scan_glucose, heart_rate, distance)

        lifelog_model = MinuteBasedLifelog(minute_id, time, geolocation, song, activity_model, wear_imgs_id, cam_imgs_id)
        minute_ids.append(minute_id)
        lifelog_models.append(lifelog_model)
    df = pd.DataFrame({'minute_id': minute_ids, 'objects': lifelog_models}, columns=['minute_id', 'objects'])
    return df


def parse_visual_concepts(df):
    result = []
    image_ids = []
    for v in df.values:
        image_id, image_path, *r = v
        attrs = r[:10]
        categories = np.asarray(r[10:20]).reshape((-1,2)).tolist()
        concepts = np.asarray(r[20:]).reshape((-1,3)).tolist()
        concepts = [Concept(x[0], x[1], x[2]) for x in concepts]
        image_ids.append(image_id)
        result.append(VisualImageConcept(image_id, image_path, attrs, categories, concepts))
    df = pd.DataFrame({'image_id':image_ids, 'objects':result}, columns=['image_id', 'objects'])
    return df


@time_this
def load_data_as_hdf(person_id):
    parent_folder = Path.cwd().parent / 'data'
    minute_based_lifelog_fold = parent_folder / 'minute_based_table'
    minute_based_lifelog_fp = minute_based_lifelog_fold / '{}.hdf'.format(person_id)
    visual_concepts_fold = parent_folder / 'visual_concepts'
    visual_concepts_fp = visual_concepts_fold / '{}_categories_attr_concepts.hdf'.format(person_id)

    if not osp.exists(str(minute_based_lifelog_fp)):
        df = load_csv_file(minute_based_lifelog_fold, '{}.csv'.format(person_id))
        df.to_hdf(minute_based_lifelog_fp, 'data', mode='w')

    if not osp.exists(str(visual_concepts_fp)):
        df = load_csv_file(visual_concepts_fold, '{}_categories_attr_concepts.csv'.format(person_id))
        df.to_hdf(visual_concepts_fp, 'data', mode='w')

    minute_based_lifelog_df = pd.read_hdf(minute_based_lifelog_fp, 'data')
    visual_concepts_df = pd.read_hdf(visual_concepts_fp, 'data')
    return minute_based_lifelog_df, visual_concepts_df
