from config import DatasetName, DatasetType
from train import Train
from train_single_path import TrainSingle
from train_online import TrainOnline
from test import Test
from AffectNetClass import AffectNet
from RafdbClass import RafDB
from FerPlusClass import FerPlus
from visualize_feature_maps import FeatureMapVisualizer

from data_helper import DataHelper

if __name__ == '__main__':
    # fm_vis = FeatureMapVisualizer(dataset_name=DatasetName.affectnet, weight_path='./3_v2_0.57.h5')
    # fm_vis.visualize()
    ''''''

    dhl = DataHelper()

    '''<><><><><><><>AffectNet<><><><><><><>'''
    affect_net = AffectNet(ds_type=DatasetType.train_7)
    '''create from the original data'''
    '''7 labels'''
    # affect_net.read_csv(ds_name=DatasetName.affectnet, ds_type=DatasetType.train_7, FLD_model_file_name='./ds_136_ef.h5', is_7=True)
    # affect_net.read_csv(ds_name=DatasetName.affectnet, ds_type=DatasetType.eval_7, FLD_model_file_name='./ds_136_ef.h5', is_7=True)
    '''8 labels'''
    # affect_net.read_csv(ds_name=DatasetName.affectnet, ds_type=DatasetType.train, FLD_model_file_name='./ds_136_ef.h5')
    # affect_net.read_csv(ds_name=DatasetName.affectnet, ds_type=DatasetType.eval, FLD_model_file_name='./ds_136_ef.h5')

    '''create synthesized landmarks'''
    # affect_net.create_synthesized_landmarks(model_file='./ds_136_ef.h5')

    '''upsampling'''
    # affect_net.upsample_data_fix_rate()
    '''create masked-img'''
    # affect_net.create_masked_image()
    '''pre-processing'''
    # affect_net.create_derivative_mask()
    # affect_net.create_au_mask()
    # affect_net.create_spatial_masks()

    # affect_net = AffectNet(ds_type=DatasetType.eval_7)
    # affect_net.create_spatial_masks()

    # affect_net.report()

    ''''''

    '''<><><><><><><>RAF-DB<><><><><><><>'''
    # train
    raf_db = RafDB(ds_type=DatasetType.train)
    # raf_db.create_from_orig(ds_type=DatasetType.train)  # here we change the labels to be consistent to AffectNet
    # test:
    # raf_db = RafDB(ds_type=DatasetType.test)
    # raf_db.create_from_orig(ds_type=DatasetType.test)  # here we change the labels to be consistent to AffectNet

    # raf_db.create_synthesized_landmarks(model_file='./ds_136_ef.h5', test_print=False)
    # # raf_db.upsample_data()
    # raf_db.upsample_data_fix_rate()
    # raf_db.create_masked_image()
    # #
    # raf_db = RafDB(ds_type=DatasetType.test)
    # raf_db.create_synthesized_landmarks(model_file='./ds_136_ef.h5', test_print=False)
    # raf_db.create_masked_image()
    ''''''
    # raf_db = RafDB(ds_type=DatasetType.train)
    # raf_db.upsample_data_fix_rate()

    # raf_db.create_spatial_masks()
    # raf_db.create_masked_image()

    # raf_db = RafDB(ds_type=DatasetType.test)
    # raf_db.create_spatial_masks()

    # raf_db = RafDB(ds_type=DatasetType.train)
    # raf_db.relabel()
    # raf_db = RafDB(ds_type=DatasetType.test)
    # raf_db.relabel()

    # raf_db = RafDB(ds_type=DatasetType.train)
    # raf_db.report(aug=False)
    #
    # raf_db = RafDB(ds_type=DatasetType.test)
    # raf_db.report()

    '''<><><><><><><>FER2013<><><><><><><>'''
    # fer_2013 = FerPlus(ds_type=DatasetType.train)
    # fer_2013.create_from_orig()  # here we change the labels to be consistent to AffectNet
    # fer_2013.create_synthesized_landmarks(model_file='./ds_136_ef.h5')
    # fer_2013.upsample_data_fix_rate()
    #
    fer_2013 = FerPlus(ds_type=DatasetType.test)
    fer_2013.create_from_orig()
    #
    # fer_2013 = RafDB(ds_type=DatasetType.train)
    # fer_2013.report(aug=False)
    #
    # fer_2013 = RafDB(ds_type=DatasetType.test)
    # fer_2013.report()


    '''<><><><><><><>SFEW<><><><><><><>'''

    """train single branch"""

    """train 3 branch model"""
    '''affectNet'''
    # trainer = Train(dataset_name=DatasetName.affectnet, ds_type=DatasetType.train_7,  lr=1e-4)
    # trainer.train(arch='mobileNetV2_5', weight_path=None)
    # trainer.train(arch='mobileNetV2_5', weight_path='./last_11_may_60.h5')

    # trainer.train(arch='mobileNetV2', weight_path=None)
    # trainer.train(arch='efn-b3', weight_path=None)
    #
    '''rafdb'''
    # trainer = Train(dataset_name=DatasetName.rafdb, ds_type=DatasetType.train, lr=1e-2)
    # trainer.train(arch='efn-b0', weight_path=None)
    # trainer.train(arch='mobileNetV2_5', weight_path=None)

    # trainer.train(arch='mobileNetV2', weight_path='./last_3_may.h5')
    # trainer.train(arch='efn-b3', weight_path=None)
    #

    """train online"""
    # trainer = TrainOnline(dataset_name=DatasetName.affectnet, ds_type=DatasetType.train_7, lr=5e-3)
    # trainer.train(arch='mobileNetV2_3', weight_path=None)

    # trainer.train(arch='mobileNetV2_3', weight_path='./ac_0.14.h5')

    # trainer = TrainOnline(dataset_name=DatasetName.rafdb, ds_type=DatasetType.train, lr=5e-3)
    # trainer.train(arch='mobileNetV2_3', weight_path=None)

    '''test'''
    # tester = Test(dataset_name=DatasetName.affectnet)
    # tester.test('/media/data2/alip/FER/affn/20_jan_2021/fer_model_88_affectnet.h5')
    # tester.test_reg('./models/fer_model_37_affectnet_0.4.h5')
