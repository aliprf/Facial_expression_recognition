class DatasetName:
    affectnet = 'affectnet'
    rafdb = 'rafdb'


class ExpressionCodesRafdb:
    Surprise = 1
    Fear = 2
    Disgust = 3
    Happiness = 4
    Sadness = 5
    Anger = 6
    Neutral = 7

class ExpressionCodesAffectnet:
    neutral = 0
    happy = 1
    sad = 2
    surprise = 3
    fear = 4
    disgust = 5
    anger = 6
    contempt = 7
    none = 8
    uncertain = 9
    noface = 10

class DatasetType:
    train = 0
    train_7 = 1
    eval = 2
    eval_7 = 3
    test = 4


class LearningConfig:
    # batch_size = 75
    batch_size = 30
    # batch_size = 3
    virtual_batch_size = 375
    epochs = 70
    # expression_output_len = 8 # we generated five classes
    embedding_size = 256


class InputDataSize:
    image_input_size = 224

class RafDBConf:
    _prefix_path = '/media/data3/ali/FER_DS/RAF-DB'  # --> zeue
    # _prefix_path = '/media/data2/alip/FER_DS/RAF-DB'  # --> Atlas
    # _prefix_path = '/media/ali/data/FER/RAF-DB'  # --> local

    orig_annotation_txt_path = _prefix_path + '/orig/RAFDB-NEED/list_patition_label.txt'
    orig_image_path = _prefix_path + '/orig/RAFDB-NEED/original/'
    orig_bounding_box = _prefix_path + '/orig/RAFDB-NEED/boundingbox/'

    '''only 7 labels'''
    no_aug_train_img_path = _prefix_path + '/train_set/images/'
    no_aug_train_annotation_path = _prefix_path + '/train_set/annotations/'

    aug_train_img_path = _prefix_path + '/train_set_aug/images/'
    aug_train_annotation_path = _prefix_path + '/train_set_aug/annotations/'
    aug_train_masked_img_path = _prefix_path + '/train_set_aug/masked_images/'

    test_img_path = _prefix_path + '/test_set/images/'
    test_annotation_path = _prefix_path + '/test_set/annotations/'
    test_masked_img_path = _prefix_path + '/test_set/masked_images/'

    augmentation_factor = 5

    weight_save_path = _prefix_path + '/weight_saving_path/'


class AffectnetConf:
    """"""
    '''atlas'''
    # sshfs user1@130.253.217.32:/home/user1/ExpressionNet/Manually_Annotated_Images /media/data2/alip/FER/affectNet/orig/images/
    '''Zeus'''
    # sshfs user1@130.253.217.32:/home/user1/ExpressionNet/Manually_Annotated_Images /media/data3/ali/FER/affectNet/orig/images/
    # orig_img_path_prefix = '/media/data2/alip/FER/affectNet/orig/images/'
    # orig_img_path_prefix = '/media/data3/ali/FER/affectNet/orig/images/'

    # orig_test_path_prefix = '/media/data3/ali/affectNet/test_set_images_cropped_not_expanded/'
    # orig_test_path_prefix = '/media/data2/alip/affectNet/test_set_images_cropped_not_expanded/'

    _prefix_path = '/media/sda/Ali/Ali_data/FER_DS/affectnet'  # --> Aq
    # _prefix_path = '/media/data3/ali/FER_DS/affectNet'  # --> zeue
    # _prefix_path = '/media/data2/alip/FER_DS/affectNet'  # --> Atlas
    # _prefix_path = '/media/ali/data/FER/affectNet'  # --> local

    orig_csv_train_path = _prefix_path + '/orig/training.csv'
    orig_csv_evaluate_path = _prefix_path + '/orig/validation.csv'
    orig_csv_test_path = '/media/data2/alip/affectNet/test_set_list.csv'

    '''8 labels'''
    no_aug_train_img_path = _prefix_path + '/train_set/images/'
    no_aug_train_annotation_path = _prefix_path + '/train_set/annotations/'

    aug_train_img_path = _prefix_path + '/train_set_aug/images/'
    aug_train_annotation_path = _prefix_path + '/train_set_aug/annotations/'
    aug_train_masked_img_path = _prefix_path + '/train_set_aug/masked_images/'

    eval_img_path = _prefix_path + '/eval_set/images/'
    eval_annotation_path = _prefix_path + '/eval_set/annotations/'
    eval_masked_img_path = _prefix_path + '/eval_set/masked_images/'

    '''7 labels'''
    no_aug_train_img_path_7 = _prefix_path + '/train_set_7/images/'
    no_aug_train_annotation_path_7 = _prefix_path + '/train_set_7/annotations/'

    aug_train_img_path_7 = _prefix_path + '/train_set_7_aug/images/'
    aug_train_annotation_path_7 = _prefix_path + '/train_set_7_aug/annotations/'
    aug_train_masked_img_path_7 = _prefix_path + '/train_set_7_aug/masked_images/'

    eval_img_path_7 = _prefix_path + '/eval_set_7/images/'
    eval_annotation_path_7 = _prefix_path + '/eval_set_7/annotations/'
    eval_masked_img_path_7 = _prefix_path + '/eval_set_7/masked_images/'

    '''------'''
    # revised_test_img_path = _prefix_path + '/test_set/images/'
    # revised_test_annotation_path = _prefix_path + '/test_set/annotations/'

    weight_save_path = _prefix_path + '/weight_saving_path/'

    num_of_samples_train = 2420940
    num_of_samples_train_7 = 0
    num_of_samples_eval = 3999




