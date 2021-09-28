from typing import List
from PIL import Image
import numpy as np
import os
import joblib
from sklearn.svm import SVC
import logging
import time

image_height = 128
image_width = 128

train_image_dir = './train'
val_image_dir = './val'
model_path = './model'
# 替换模型修改这个↓
model_name = 'icon_model'

class icon_classify:
    def __init__(self, logger, retrain = False, model_save_name = 'icon_model_test', show_val = True):
        self.logger = logger
        logger.info(f'------------------------------------------')
        logger.info(f'Init succeed! retrain={retrain}, model_save_name={model_save_name}, show_val={show_val}')

        if not retrain:
            logger.info("Load model...")
            start_time = time.time()

            self.svm = joblib.load(os.path.join(model_path, model_name))

            logger.info("Load over!")
            logger.info(f"Cost time: {time.time() - start_time}")
        else:
            logger.info("Start retrain...")
            start_time = time.time()

            self.svm = SVC(verbose = 1)
            train_x, train_y = self.load_train_data()

            logger.info("Start fit, it may cost 1-30 minutes...")
            self.svm.fit(train_x, train_y)

            logger.info("Retrain over!")
            logger.info(f"Cost time: {time.time() - start_time}")
                        
            joblib.dump(self.svm, os.path.join(model_path, model_save_name))

        if show_val:
            self.val_and_show()


    def load_train_data(self):
        self.logger.info("Start load train data, it may cost 1-60 seconds...")
        train_x = []
        train_y = []

        start_time = time.time()
        train_list = os.listdir(train_image_dir)
        for icon_name in train_list:
            icon_path = os.path.join(train_image_dir, icon_name)

            img = Image.open(icon_path)
            img_np = np.array(img)
            train_x.append(img_np.reshape(image_height*image_height))
            train_y.append(icon_name.split('_')[0])

        self.logger.info(f"Load over! Total {len(train_y)} samples.")
        self.logger.info(f"Cost time: {time.time() - start_time}")

        return np.array(train_x), np.array(train_y)


    def image_classify(self, image_list) -> List[str]:
        # image_list should be the list of your images

        np_array = np.zeros((len(image_list), image_height*image_height))
        for i, image in enumerate(image_list):
            image = image.resize((image_height,image_height),Image.ANTIALIAS)
            out = image.convert('1')
            np_array[i] = np.array(out).reshape(image_height*image_height)
        
        return self.np_classify(np_array)


    def np_classify(self, np_array) -> List[str]:
        # np should be dim 2
        # per image be image_height * image_width
        # np_list should be (n, image_height * image_width(default 16384))
        # n is the sum of your images

        assert np_array.ndim == 2, 'np should be dim 2' 
        assert np_array.shape[1] == image_height * image_width, "image size doesn't fit"
        
        res = self.svm.predict(np_array)
        return res


    def val_and_show(self):
        self.logger.info("Start load val data...")
        val_x = []
        val_y = []

        start_time = time.time()
        val_list = os.listdir(val_image_dir)
        for icon_name in val_list:
            icon_path = os.path.join(val_image_dir, icon_name)

            img = Image.open(icon_path)
            img_np = np.array(img)
            val_x.append(img_np.reshape(image_height*image_height))
            val_y.append(icon_name.split('_')[0])

        self.logger.info(f"Load over! Total {len(val_y)} samples.")
        self.logger.info(f"Cost time: {time.time() - start_time}")

        self.logger.info("Start classify...")
        start_time = time.time()

        result = self.np_classify(np.array(val_x))

        self.logger.info(f"classify over!")
        self.logger.info(f"Cost time: {time.time() - start_time}")

        total_cnt = 0
        true_cnt = 0
        for icon_name, res in zip(val_list, result):
            total_cnt += 1
            if icon_name.split('_')[0] != res:
                self.logger.debug(f'{icon_name}\t{res}')
            else:
                true_cnt += 1
        self.logger.info(f'{true_cnt}/{total_cnt}={true_cnt/total_cnt}')

        
