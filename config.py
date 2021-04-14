class DatasetName:
    affectnet = 'affectnet'


class ExpressionCodesAffectnet:
    '''this is for test. NEVER use it'''
    # neutral = 1
    # happy = 2
    # sad = 3
    # surprise = 4
    # fear = 5
    # disgust = 6
    # anger = 7
    # contempt = 8
    # none = 9
    # uncertain = 10
    # noface = 11

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


class LearningConfig:
    batch_size = 25
    # batch_size = 5
    epochs = 200
    velocity_output_len = 6 # we generated five classes
    # expression_output_len = 8 # we generated five classes
    expression_output_len_shrunk = 4 # we generated five classes
    embedding_size = 256


class InputDataSize:
    image_input_size = 224

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

    _prefix_path = 'media/data3/ali/FER_DS/affectNet'  # --> zeue
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
    eval_img_path = _prefix_path + '/eval_set/images/'
    eval_annotation_path = _prefix_path + '/eval_set/annotations/'

    '''7 labels'''
    no_aug_train_img_path_7 = _prefix_path + '/train_set_7/images/'
    no_aug_train_annotation_path_7 = _prefix_path + '/train_set_7/annotations/'

    aug_train_img_path_7 = _prefix_path + '/train_set_7_aug/images/'
    aug_train_annotation_path_7 = _prefix_path + '/train_set_7_aug/annotations/'
    eval_img_path_7 = _prefix_path + '/eval_set_7/images/'
    eval_annotation_path_7 = _prefix_path + '/eval_set_7/annotations/'

    '''------'''
    # revised_test_img_path = _prefix_path + '/test_set/images/'
    # revised_test_annotation_path = _prefix_path + '/test_set/annotations/'

    augmentation_factor = 2




