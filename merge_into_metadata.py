import os
import os.path as osp
from pathlib import Path
import pandas as pd
from collections import defaultdict
import math
import itertools
import datetime
import json
from dateutil import tz
from tzwhere import tzwhere


parent_folder = Path.cwd().parent
lifelog_object_folder = parent_folder / 'LifelogObject'
lifelog_place_folder = parent_folder / 'LifelogPlacesLog'
lifelog_hr_folder = parent_folder / 'LifelogHeartRate'
lifelog_calories_folder = parent_folder / 'LifelogCalories'
lifelog_distance_folder = parent_folder / 'LifelogDistances'
lifelog_steps_folder = parent_folder / 'LifelogSteps'
minute_based_folder = parent_folder / 'minute_based' 
activity_folder = minute_based_folder / 'activity'
place_folder = minute_based_folder / 'place'


def create_visual_concept_metadata():
    vs = []
    date = sorted(os.listdir(str(lifelog_object_folder)))
    count = 1
    for _date in date:
        if _date == '.DS_Store':
            continue
        object_date_path = lifelog_object_folder / _date
        place_cat_date_path = lifelog_place_folder / 'category_logs' / _date
        place_attr_date_path = lifelog_place_folder / 'attribute_logs' / _date
        img_objs = sorted(os.listdir(str(object_date_path)))
        img_cats = sorted(os.listdir(str(place_cat_date_path)))
        img_attr = sorted(os.listdir(str(place_attr_date_path)))
        assert len(img_objs) == len(img_cats) and len(img_attr) == len(img_cats)
        cnt = defaultdict(int)
        num_imgs = len(img_objs)
        for i in range(num_imgs):
            if img_objs[i] == '.DS_Store':
                continue
            print(count)
            count += 1
            _, _, _d, _time = img_objs[i].split('_')
            minute_id = int(float(_time[:2]) * 60 + float(_time[2:4]))
            _count = cnt[minute_id]
            cnt[minute_id] += 1
            image_id = 'u1_{}_{:02d}{:02d}_i{:02d}'.format(_d, minute_id // 60, minute_id % 60, _count)
            image_path = '{}/{}'.format(_date, img_objs[i].split('.')[0] + '.JPG')
            attr_df = pd.read_csv(str(place_attr_date_path / img_attr[i]), header=None)
            cat_df = pd.read_csv(str(place_cat_date_path / img_cats[i]), header=None)
            try:
                obj_df = pd.read_csv(str(object_date_path / img_objs[i]), header=None)
                obj_df = obj_df.values.tolist()
                obj_df.append([None, None, None] * (25 - len(obj_df)))
                obj_df = list(itertools.chain(*obj_df))
            except Exception as e:
                print(e)
                obj_df = [None] * 75
            vs.append(tuple([image_id, image_path, *attr_df.values.tolist()[0], *list(itertools.chain(*cat_df.values.tolist())), *obj_df]))
    cols = ['image_id', 'image_path', *['attribute_top{:02d}'.format(j+1) for j in range(10)]]
    for i in range(5):
        cols += ['category_top{:02d}'.format(i+1), 'category_top{:02d}_score'.format(i+1)]
    for i in range(25):
        cols += ['concept_class_top{:02d}'.format(i+1), 'concept_score_top{:02d}'.format(i+1), 'concept_bbox_top{:02d}'.format(i+1)]
    df = pd.DataFrame(data=vs, columns=cols)
    df.to_csv('u1_categories_attr_concepts.csv', index=False, na_rep='NULL')


def round_to_minute(dt):
    # if dt.second > 0:
    #     dt += datetime.timedelta(minutes=1)
        # dt -= datetime.timedelta(seconds=dt.second)
    dt -= datetime.timedelta(seconds=dt.second)
    return dt


def parse_activity(activities, activity):
    for v in activity.values:
        _, act_name, _, start, end, *r = v
        start = round_to_minute(datetime.datetime.strptime(start, '%Y-%m-%dT%H:%M:%S%z').astimezone(tz.gettz('UTC')))
        end = round_to_minute(datetime.datetime.strptime(end, '%Y-%m-%dT%H:%M:%S%z').astimezone(tz.gettz('UTC')))
        delta = datetime.timedelta(minutes=1)
        now = start 
        while now <= end:
            _date = '{:04d}{:02d}{:02d}'.format(now.year, now.month, now.day)
            _time = '{:02d}{:02d}'.format(now.hour, now.minute)
            activities[_date][_time] = act_name
            now += delta


def parse_place(places, place):
    for v in place.values:
        _, place_name, start, end, _, lat, lng, *r = v

        start = round_to_minute(datetime.datetime.strptime(start, '%Y-%m-%dT%H:%M:%S%z').astimezone(tz.gettz('UTC')))
        end = round_to_minute(datetime.datetime.strptime(end, '%Y-%m-%dT%H:%M:%S%z').astimezone(tz.gettz('UTC')))
        delta = datetime.timedelta(minutes=1)
        now = start
        while now <= end:
            _date = '{:04d}{:02d}{:02d}'.format(now.year, now.month, now.day)
            _time = '{:02d}{:02d}'.format(now.hour, now.minute)
            places[_date][_time] = (place_name, lat, lng)
            now += delta


def create_minute_based_metadata():
    minute_based_metadata = []
    date = sorted(os.listdir(str(lifelog_object_folder)))
    image_ids = pd.read_csv('u1_categories_attr_concepts.csv', keep_default_na=False)['image_id']
    tzw = tzwhere.tzwhere()
    activities = defaultdict(dict)
    places = defaultdict(dict)

    for _date in [*date, '2018_06_06']:
        if _date == '.DS_Store':
            continue
        activity = pd.read_csv(str(activity_folder / 'activities_{}.csv'.format(_date.replace('_', ''))))
        parse_activity(activities, activity)
        place = pd.read_csv(str(place_folder / 'places_{}.csv'.format(_date.replace('_', ''))))
        parse_place(places, place)

    for _date in date:
        if _date == '.DS_Store':
            continue
        calories = json.load(open(str(lifelog_calories_folder / '{}.json'.format(_date))))['activities-calories-intraday']['dataset']
        heart_rates = json.load(open(str(lifelog_hr_folder / '{}.json'.format(_date))))['activities-heart-intraday']['dataset']
        steps = json.load(open(str(lifelog_steps_folder / '{}.json'.format(_date))))['activities-steps-intraday']['dataset']
        distances = json.load(open(str(lifelog_distance_folder / '{}.json'.format(_date))))['activities-distance-intraday']['dataset']
        current_timezone = ''
        num_minutes = 1440
        cnt = [0] * 4
        for i in range(num_minutes):
            _time = '{:02d}:{:02d}:00'.format(i // 60, i % 60)
            datetime_pattern = '{}_{:02d}{:02d}'.format(_date.replace('_', ''), i // 60, i % 60)
            imgid_by_datetime = image_ids[image_ids.str.contains(datetime_pattern)].tolist()
            imgid_by_datetime = [imgid_by_datetime, [None] * (20 - len(imgid_by_datetime))]
            imgid_by_datetime = list(itertools.chain(*imgid_by_datetime))
            camid_by_datetime = [None] * 15
            if cnt[0] < len(calories) and calories[cnt[0]]['time'] == _time:
                calory = calories[cnt[0]]['value']
                cnt[0] += 1
            if cnt[1] < len(heart_rates) and heart_rates[cnt[1]]['time'] == _time:
                heart_rate = heart_rates[cnt[1]]['value']
                cnt[1] += 1
            if cnt[2] < len(steps) and steps[cnt[2]]['time'] == _time:
                step = steps[cnt[2]]['value']
                cnt[2] += 1
            if cnt[3] < len(distances) and distances[cnt[3]]['time'] == _time:
                distance = distances[cnt[3]]['value']
                cnt[3] += 1
            try:
                activity = activities[_date.replace('_', '')]['{:02d}{:02d}'.format(i // 60, i % 60)]
            except Exception as ex:
                print('Activity', ex)
                activity = None
            try:
                location_name, lat, lng = places[_date.replace('_', '')]['{:02d}{:02d}'.format(i // 60, i % 60)]
            except Exception as ex:
                print('Place', ex)
                location_name, lat, lng = None, None, None
            if lat != None and lng != None:
                if tzw.tzNameAt(lat, lng) != None:
                    current_timezone = tzw.tzNameAt(lat, lng)
            minute_id = 'u1_{}_{:02d}{:02d}'.format(_date.replace('_', ''), i // 60, i % 60)
            utc_time = '{}_{:02d}{:02d}_UTC'.format(_date.replace('_', ''), i // 60, i % 60)
            year, month, day = _date.split('_')
            utc_time_extra = datetime.datetime(int(year), int(month), int(day), hour=i // 60, minute=i % 60)
            local_time = utc_time_extra.replace(tzinfo=tz.gettz('UTC')).astimezone(tz.gettz(current_timezone))
            local_time = '{:04d}{:02d}{:02d}_{:02d}{:02d}'.format(local_time.year, local_time.month, local_time.day, local_time.hour, local_time.minute)
            minute_based_metadata.append(tuple([minute_id, utc_time, local_time, current_timezone, lat, lng, location_name, None, activity, step, calory, None, None, heart_rate, distance,
                *imgid_by_datetime, *camid_by_datetime]))
    cols = ['minute_ID', 'utc_time', 'local_time', 'time_zone', 'lat', 'lon', 'name', 'song', 'activity', 'steps', 'calories', 'historic_glucose', 'scan_glucose',
        'heart_rate', 'distance', *['img{:02d}_id'.format(i) for i in range(20)], *['cam{:02d}_id'.format(i) for i in range(15)]]
    df = pd.DataFrame(data=minute_based_metadata, columns=cols)
    df.to_csv('u1.csv', index=False, na_rep='NULL')


if __name__ == '__main__':
    create_visual_concept_metadata()
    create_minute_based_metadata()