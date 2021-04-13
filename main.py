from config import DatasetName, DatasetType, AffectnetConf
from train import Train
from test import Test
from AffectNetClass import  AffectNet
from data_helper import DataHelper
if __name__ == '__main__':

    dhl = DataHelper()

    '''<><><><><><><>AffectNet<><><><><><><>'''
    affect_net = AffectNet(ds_type=DatasetType.train)
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
    affect_net.upsample_data()

    '''pre-processing'''
    # affect_net.create_derivative_mask()
    # affect_net.create_au_mask()
    # affect_net.create_spatial_masks()
    ''''''


    '''<><><><><><><>RAF-DB<><><><><><><>'''
    '''<><><><><><><>FERPLUS<><><><><><><>'''
    '''<><><><><><><>SFEW<><><><><><><>'''

    """train"""
    trainer = Train(dataset_name=DatasetName.affectnet, ds_type=DatasetType.train)
    trainer.train(arch='mobileNetV2', weight_path=None)


    # trainer.train(arch='efficientNet', weight_path='/media/data2/alip/FER/affn/20_jan_2021/fer_model_88_affectnet.h5',
    #               mode=0) # 0-> categorical, 1->regression

    # trainer.train(arch='mobileNetV2', weight_path=None)

    '''test'''
    # tester = Test(dataset_name=DatasetName.affectnet)
    # tester.test('/media/data2/alip/FER/affn/20_jan_2021/fer_model_88_affectnet.h5')
    # tester.test_reg('./models/fer_model_37_affectnet_0.4.h5')