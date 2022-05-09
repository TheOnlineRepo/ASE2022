# encoding=utf-8
import sys
sys.path.append('/home/zhiyu/DeepSuite/adversarial')
import tensorflow as tf
from adv_function import *
import time
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.disable_eager_execution()


def main():
    _, x_train, y_train, x_test, y_test = load_model_and_testcase(args.path1, args.model, args.dataset)
    base_dir = os.path.join(args.path0, 'FGSM_RQ3')
    fitness_list = ['None', 'nbc', 'kmnc', 'nc', 'snac', 'tknc']
    ori_result = []
    adv_result = []
    for fitness in fitness_list:
        model_path = os.path.join(args.path0, 'adv_retrain_models', args.model, fitness+'.h5')
        try:
            model = load_model(model_path)
        except:
            continue
        predictions = np.argmax(model.predict(x_test), axis=1)
        wrong_pred_id = np.where(predictions != y_test)[0]
        accuracy = (len(y_test) - len(wrong_pred_id)) / len(y_test)
        print("Accuracy on origin test examples: {}%".format(accuracy * 100))
        ori_result.append(accuracy)

        adv_train_x, adv_train_y, accuracy = attack_mathod_generater(model, 'FGSM', x_test, y_test, args.adv_eps)
        save_dir = os.path.join(base_dir, args.model, fitness)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'adv_test_samples.npy')
        np.savez(save_path, x=adv_train_x, y=adv_train_y)
        adv_result.append(accuracy)
    print(fitness_list)
    print(ori_result)
    print(adv_result)


if __name__ == '__main__':
    start_time = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', help='dataset is either mnist or cifar10', type=str,
                        default='mnist')
    parser.add_argument('-model', help='model of mnist is leNet_1/leNet_4/leNet_5/resnet20/vgg16', type=str,
                        default='leNet_1')
    parser.add_argument('-path0', type=str, default='/media/data0/DeepSuite')
    parser.add_argument('-path1', type=str, default='/media/data1/DeepSuite')
    parser.add_argument('-adv_eps', type=float, default=0.1)

    args = parser.parse_args()
    print(args)
    main()



