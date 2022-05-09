# encoding=utf-8
import sys

sys.path.append('/home/zhiyu/DeepSuite/adversarial')
import tensorflow as tf
from adv_function import *
from empirical_study.RQ3 import PGD_deepXplore
import time
import argparse
from empirical_study.function import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.disable_eager_execution()


def gen_and_save(model, origin_x, origin_y, adv_eps, model_name):
    classifier = KerasClassifier(model=model, use_logits=False, clip_values=[0, 1])
    attack = PGD_deepXplore.ProjectedGradientDescent_fitness(estimator=classifier, eps=adv_eps, eps_step=adv_eps / 3)
    x_test_adv = attack.generate(x=origin_x, choose_num=10, fit_cot=0.5, fitness=args.fitness, model=model,
                                 model_name=model_name)
    predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
    wrong_pred_id = np.where(predictions != origin_y)[0]
    accuracy = (len(origin_y) - len(wrong_pred_id)) / len(origin_y)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
    return x_test_adv


def main():
    choose_num = 10
    model, x_train, y_train, x_test, y_test = load_model_and_testcase(args.path, args.model, args.dataset)
    print(model.summary)

    save_dir = os.path.join(args.path, 'PGD_DeepXplore_2layer', args.model, args.fitness)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_grad = get_cov_grad(x_train, model, args.fitness, choose_num, args.model, int(args.layer))
    train_x_path = os.path.join(save_dir, 'train_con_grad.npy')
    np.save(train_x_path, train_grad)

    test_grad = get_cov_grad(x_test, model, args.fitness, choose_num, args.model, int(args.layer))
    test_x_path = os.path.join(save_dir, args.layer + '_test_con_grad.npy')
    np.save(test_x_path, test_grad)

    # np.save(train_y_path, y_train)

    # adv_x_test = gen_and_save(model, x_test, y_test, adv_eps, args.model)
    # test_x_path = os.path.join(save_dir, 'test_x.npy')
    # test_y_path = os.path.join(save_dir, 'test_y.npy')
    # np.save(test_x_path, adv_x_test)
    # np.save(test_y_path, y_test)


if __name__ == '__main__':
    start_time = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', help='dataset is either mnist or cifar10', type=str, default='cifar10')
    parser.add_argument('-model', help='model of mnist is leNet_1/leNet_4/leNet_5/resnet20/vgg16', type=str,
                        default='resnet20_cifar10')
    parser.add_argument('-path', help='directory where models and datasets are stored', type=str,
                        default='/media/data1/DeepSuite')
    parser.add_argument('-test_size', type=float, default=0.1)
    parser.add_argument('-fitness', type=str, default='nc')
    parser.add_argument('-layer', type=int, default=2)
    parser.add_argument('-GPU', type=int, default=0)
    parser.add_argument('-attack_eps_id', type=int, default=0)

    args = parser.parse_args()
    print(args)
    main()
