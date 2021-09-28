import collections
from typing import *
from PIL import Image
import numpy as np
import os
import joblib
from numpy.core.records import array
from sklearn.svm import SVC
import logging
import time


image_height = 128
image_width = 128

train_image_dir = './train'
val_image_dir = './val'
model_path = './model'
# 替换模型修改这个↓
model_default_save_dir = 'angle_model'

class_list = [
    "ACLineEnd",
    "Bus",
    "Capacitor",
    "CBreaker",
    "Disconnector",
    "Generator",
    "GroundDisconnector",
    "Reactor",
    "SVG",
    "Transformer2",
    "Transformer3",
]


class angle_classify:
    def __init__(self, logger = None, retrain = False, model_save_dir = 'angel_model_test', show_val = True):
        self.logger = logger if logger else logging.getLogger()
        logger.info(f'------------------------------------------')
        logger.info(f'Init succeed! retrain={retrain}, model_save_dir={model_save_dir}, show_val={show_val}')

        if not retrain:
            logger.info("Load model...")
            start_time = time.time()

            self.load_model()
            
            logger.info("Load over!")
            logger.info(f"Cost time: {time.time() - start_time}")
        else:
            model_dir_path = os.path.join(model_path, model_save_dir)
            assert os.path.exists(model_dir_path), "No such dir!"
            logger.info("Start retrain, it may cost 1-30 minutes...")
            start_time = time.time()

            self.svm = {}
            self.retrain_svm(model_dir_path)

            logger.info("Retrain over!")
            logger.info(f"Cost time: {time.time() - start_time}")

        if show_val:
            self.val_and_show()


    def load_model(self):
        self.svm = {}
        model_dir_path = os.path.join(model_path, model_default_save_dir)
        for class_name in class_list:
            class_model_path = os.path.join(model_dir_path, class_name)
            self.svm[class_name] = joblib.load(class_model_path)
            self.logger.info(f"Load {class_name} model succeed!")


    def retrain_svm(self, model_save_dir):
        train_x, train_y = self.load_train_data()
        for class_name in class_list:
            self.logger.info(f"Retraining {class_name} model...")

            self.svm[class_name] = SVC()
            self.svm[class_name].fit(train_x[class_name], train_y[class_name])
            joblib.dump(self.svm[class_name], os.path.join(model_save_dir, class_name))


    def load_train_data(self) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[str]]]:
        self.logger.info(f"Start load train data...")
        train_x = collections.defaultdict(list)
        train_y = collections.defaultdict(list)

        start_time = time.time()
        train_list = os.listdir(train_image_dir)
        for icon_name in train_list:
            class_name = icon_name.split('_')[0]
            ground_angle = icon_name.split('_')[2].split('.')[0]
            icon_path = os.path.join(train_image_dir, icon_name)

            img = Image.open(icon_path)
            img_np = np.array(img)
            train_x[class_name].append(img_np.reshape(image_height*image_height))
            train_y[class_name].append(ground_angle)

        self.logger.info(f"Load over! Total {sum([len(item) for item in train_y.values()])} samples.")
        self.logger.info(f"Cost time: {time.time() - start_time}")

        return train_x, train_y


    def image_classify(self, image_list : Dict[str, List[Image.Image]]) -> Dict[str, List[str]]:
        # image_list should be the list of your images_class
        # input images should be divided by class

        np_array_dict = {}
        for class_name in class_list:
            np_array = np.zeros((len(image_list[class_name]), image_height*image_height))
            for i, image in enumerate(image_list[class_name]):
                image = image.resize((image_height,image_height),Image.ANTIALIAS)
                out = image.convert('1')
                np_array[i] = np.array(out).reshape(image_height*image_height)
            np_array_dict[class_name] = np_array
        return self.np_classify(np_array_dict)



    def np_classify(self, np_array_dict) -> Dict[str, List[str]]:
        # np should be dim 2
        # per image be image_height * image_width
        # np_list should be (n, image_height * image_width(default 16384))
        # n is the sum of your images

        result = {}
        for class_name, np_array in np_array_dict.items():
            result[class_name] = self.svm[class_name].predict(np_array)
        return result


    def val_and_show(self):
        self.logger.info("Start load val data...")
        val_x = collections.defaultdict(list)
        val_y = collections.defaultdict(list)

        start_time = time.time()
        val_list = os.listdir(val_image_dir)
        for icon_name in val_list:
            class_name = icon_name.split('_')[0]
            icon_path = os.path.join(val_image_dir, icon_name)

            img = Image.open(icon_path)
            img_np = np.array(img)
            val_x[class_name].append(img_np.reshape(image_height*image_height))
            val_y[class_name].append(icon_name)

        self.logger.info(f"Load over! Total {sum([len(item) for item in val_y.values()])} samples.")
        self.logger.info(f"Cost time: {time.time() - start_time}")

        self.logger.info("Start classify...")
        start_time = time.time()

        result = self.np_classify(val_x)

        self.logger.info(f"classify over!")
        self.logger.info(f"Cost time: {time.time() - start_time}")

        total_cnt = 0
        true_cnt = 0
        for class_name in class_list:
            if class_name in val_y:
                class_total_cnt = 0
                class_true_cnt = 0
                for icon_name, res in zip(val_y[class_name], result[class_name]):
                    class_total_cnt += 1
                    if icon_name.split('_')[2].split('.')[0] != res:
                        self.logger.debug(f'{icon_name}\t{res}')
                    else:
                        class_true_cnt += 1
                self.logger.info(f'{class_name}: {class_true_cnt}/{class_total_cnt}={class_true_cnt/class_total_cnt}')
                total_cnt += class_total_cnt
                true_cnt += class_true_cnt
        self.logger.info(f'Total result: {true_cnt}/{total_cnt}={true_cnt/total_cnt}')
        
