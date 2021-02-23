class DatasetName:
    affectnet = 'affectnet'


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
    eval = 1
    test = 2


class LearningConfig:
    batch_size = 25
    # batch_size = 5
    epochs = 200
    velocity_output_len = 6 # we generated five classes
    # expression_output_len = 8 # we generated five classes
    expression_output_len_shrunk = 4 # we generated five classes


class InputDataSize:
    image_input_size = 224


class AffectnetConf:
    _prefix_path = '/media/data2/alip/FER/affectNet'  # --> Atlas
    # _prefix_path = '/media/ali/data/FER/affectNet'  # --> local

    orig_img_path_prefix = '/media/data2/alip/FER/affectNet/orig/images/'
    orig_test_path_prefix = '/media/data2/alip/affectNet/test_set_images_cropped_not_expanded/'

    orig_csv_train_path = _prefix_path + '/orig/training.csv'
    orig_csv_evaluate_path = _prefix_path + '/orig/validation.csv'
    orig_csv_test_path = '/media/data2/alip/affectNet/test_set_list.csv'

    revised_train_img_path = _prefix_path + '/train_set/images/'
    revised_train_annotation_path = _prefix_path + '/train_set/annotations/'

    revised_test_img_path = _prefix_path + '/test_set/images/'
    revised_test_annotation_path = _prefix_path + '/test_set/annotations/'

    revised_eval_img_path = _prefix_path + '/eval_set/images/'
    revised_eval_annotation_path = _prefix_path + '/eval_set/annotations/'



