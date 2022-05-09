# encoding=utf-8
import tensorflow as tf
import sys
sys.path.append('/home/zhiyu/DeepSuite/adversarial')
from adv_function import *
import time
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.disable_eager_execution()


def main():
    # adv_method_list = ['FGSM', 'PGD', 'BIM', 'DF', 'CW_Linf']
    # adv_method_list = ['FGSM', 'PGD', 'BIM', 'CW_Linf']
    # adv_method_list = ['DF', 'CW_Linf']
    adv_method_list = ['CW_Linf']
    model, x_train, y_train, x_test, y_test = load_model_and_testcase(args.path, args.model, args.dataset)
    print(model.summary)
    save_model_dir = create_adv_dataset_dir(args.save_path, args.dataset, args.model)
    train_id, valid_id, test_id = load_data_split(save_model_dir, model, x_train, y_train, x_test, y_test, args.test_size)
    # eps_list_1 = [0.01,  0.005, 0.001]
    eps_list_1 = 0.3
    eps_list_2 = 1e-4
    # eps_list_2 = [0.1, 0.2, 0.3]
    for adv_method in adv_method_list:
        print("——————", adv_method, "——————")
        if adv_method in ['BIM', 'FGSM', 'PGD', 'CW_Linf', 'CW_L2']:
            adv_eps = eps_list_1
        else:
            adv_eps = eps_list_2

        adv_save_dir = os.path.join(save_model_dir, adv_method, str(adv_eps))
        if not os.path.exists(adv_save_dir):
            os.makedirs(adv_save_dir)

        # train_save_path = os.path.join(adv_save_dir, 'train.npy.npz')
        # if not os.path.exists(train_save_path):
        #     adv_train_x, adv_train_y = attack_mathod_generater(model, adv_method, x_train[train_id], y_train[train_id], adv_eps)
        #     np.savez(train_save_path, x=adv_train_x, y=adv_train_y)
        # else:
        #     train_adv = np.load(train_save_path)
        #     adv_train_x, adv_train_y = train_adv['x'], train_adv['y']
        # accuracy = check_acc(model, adv_train_x, adv_train_y)

        valid_save_path = os.path.join(adv_save_dir, 'valid.npy.npz')
        if not os.path.exists(valid_save_path):
            adv_valid_x, adv_valid_y = attack_mathod_generater(model, adv_method, x_train[valid_id], y_train[valid_id], adv_eps)
            np.savez(valid_save_path, x=adv_valid_x, y=adv_valid_y)
        else:
            valid_adv = np.load(valid_save_path)
            adv_valid_x, adv_valid_y = valid_adv['x'], valid_adv['y']
        accuracy = check_acc(model, adv_valid_x, adv_valid_y)

        # test_save_path = os.path.join(adv_save_dir, 'test.npy.npz')
        # if not os.path.exists(test_save_path):
        #     adv_test_x, adv_test_y = attack_mathod_generater(model, adv_method, x_test[test_id], y_test[test_id], adv_eps)
        #     np.savez(test_save_path, x=adv_test_x, y=adv_test_y)
        # else:
        #     test_adv = np.load(test_save_path)
        #     adv_test_x, adv_test_y = test_adv['x'], test_adv['y']
        # accuracy = check_acc(model, adv_test_x, adv_test_y)


if __name__ == '__main__':
    start_time = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', help='dataset is either mnist or cifar10', type=str,
                        default='mnist')
    parser.add_argument('-model', help='model of mnist is leNet_1/leNet_4/leNet_5/resnet20/vgg16', type=str,
                        default='leNet_1')
    parser.add_argument('-path', help='directory where models and datasets are stored', type=str, default='/media/data1/DeepSuite')
    parser.add_argument('-save_path', type=str, default='/media/data0/DeepSuite')
    parser.add_argument('-test_size', type=float, default=0.1)
    # parser.add_argument('-adv_method', type=str,
    #                     choices=['PGD', 'FGSM', 'BIM', 'DF', 'CW', 'JSMA'], default='FGSM')
    parser.add_argument('-attack_eps_id', type=int, default=0)

    args = parser.parse_args()
    print(args)
    main()



