from config import DatasetName, DatasetType, AffectnetConf
from train import Train
from test import Test
from data_helper import DataHelper
if __name__ == '__main__':
    """pre-process data:"""
    dhl = DataHelper()
    # dhl.read_csv(ds_name=DatasetName.affectnet, ds_type=DatasetType.train, FLD_model_file_name='./ds_136_ef.h5')
    # dhl.create_mean_faces(img_path=AffectnetConf.revised_train_img_path, anno_path=AffectnetConf.revised_train_annotation_path)

    """train"""
    # trainer = Train(dataset_name=DatasetName.affectnet)
    # trainer.train(arch='efficientNet', weight_path=None, mode=0) # 0-> categorical, 1->regression
    # trainer.train(arch='mobileNetV2', weight_path=None)

    tester = Test(dataset_name=DatasetName.affectnet)

    tester.test('/media/data2/alip/FER/affn/20_jan_2021/fer_model_88_affectnet.h5')
    # tester.test_reg('./models/fer_model_37_affectnet_0.4.h5')