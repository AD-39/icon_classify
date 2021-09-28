import collections
from icon_classify import icon_classify
from angle_classify import angle_classify
import os
from PIL import Image
import logging

logger = logging.getLogger()
logger.setLevel(level = logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')

handler = logging.FileHandler("./log/log.txt")
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

sample_dir = './sample'


# icon分类
def test_icon_classify():
    samples = os.listdir(sample_dir)
    samples_img = []
    for sample in samples:
        sample_path = os.path.join(sample_dir, sample)

        samples_img.append(Image.open(sample_path))

    icon_c = icon_classify(logger, retrain=True, model_save_name='icon_model_test', show_val=True)
    icon_result = icon_c.image_classify(samples_img)

    total_cnt = 0
    true_cnt = 0
    for sample, res in zip(samples, icon_result):
        total_cnt += 1
        if sample.split('_')[0] != res:
            print(f"{sample}\t{res}")
        else:
            true_cnt += 1
    print(f'{true_cnt}/{total_cnt}={true_cnt/total_cnt}')

# 角度分类
def test_angle_classify():
    samples = os.listdir(sample_dir)
    samples_img = collections.defaultdict(list)
    for sample in samples:
        class_name = sample.split('_')[0]
        sample_path = os.path.join(sample_dir, sample)

        samples_img[class_name].append(Image.open(sample_path))

    angle_c = angle_classify(logger, retrain=False, model_save_dir='angle_model_test', show_val=False)
    angle_result = angle_c.image_classify(samples_img)

    total_cnt = 0
    true_cnt = 0
    for res_list in angle_result.values():
        for res in res_list:
            if '0' == res:
                true_cnt += 1
            total_cnt += 1
    print(f'{true_cnt}/{total_cnt}={true_cnt/total_cnt}')


# test_icon_classify()
test_angle_classify()
