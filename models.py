class MinuteBasedLifelog:
    def __init__(self, minute_id, time, geolocation, song, activity, wear_imgs, cam_imgs):
        self.minute_id = minute_id
        self.time = time
        self.geolocation = geolocation
        self.song = song
        self.activity = activity
        self.wear_imgs = wear_imgs
        self.cam_imgs = cam_imgs


class Time:
    def __init__(self, utc_time, local_time, time_zone):
        self.utc_time = utc_time
        self.local_time = local_time
        self.time_zone = time_zone


class Geolocation:
    def __init__(self, lat, lng, location_name):
        self.lat = lat
        self.lng = lng
        self.location_name = location_name


class Activity:
    def __init__(self, activity_name, steps, calories, historic_glucose, scan_glucose, heart_rate, distance):
        self.activity_name = activity_name
        self.steps = steps
        self.calories = calories
        self.historic_glucose = historic_glucose
        self.scan_glucose = scan_glucose
        self.heart_rate = heart_rate
        self.distance = distance


class VisualImageConcept:
    def __init__(self, img_id, img_path, attributes, categories, concepts):
        self.img_id = img_id
        self.img_path = img_path
        self.attributes = attributes
        self.categories = categories
        self.concepts = concepts


class Concept:
    def __init__(self, class_name, score, bbox):
        self.class_name = class_name
        self.score = score
        if bbox == ' NULL':
            bbox = '0 0 0 0'
        self.bbox = BoundingBox(bbox)


class BoundingBox:
    def __init__(self, bbox):
        top_left_x, top_left_y, bot_right_x, bot_right_y = bbox.split()
        self.top_left_x = int(float(top_left_x))
        self.top_left_y = int(float(top_left_y))
        self.bot_right_x = int(float(bot_right_x))
        self.bot_right_y = int(float(bot_right_y))
