from art.attacks.evasion import ProjectedGradientDescent, \
    FastGradientMethod, BasicIterativeMethod, DeepFool, \
    CarliniL2Method, CarliniLInfMethod, SaliencyMapMethod
from art.estimators.classification import KerasClassifier
import numpy as np


def attack_mathod_generater(model, adv_method, origin_x, origin_y, attack_eps=0.3):
    classifier = KerasClassifier(model=model, use_logits=False, clip_values=[0, 1])
    adv_method_map = {'PGD': ProjectedGradientDescent(estimator=classifier, eps=attack_eps, eps_step=attack_eps/3),
                      'FGSM': FastGradientMethod(estimator=classifier, eps=attack_eps),
                      'BIM': BasicIterativeMethod(estimator=classifier, eps=attack_eps, eps_step=attack_eps/3),
                      'DF': DeepFool(classifier=classifier, epsilon=attack_eps),
                      'CW_L2': CarliniL2Method(classifier=classifier),
                      'CW_Linf': CarliniLInfMethod(classifier=classifier, eps=attack_eps),
                      'JSMA': SaliencyMapMethod(classifier=classifier)}
    attack = adv_method_map[adv_method]
    x_test_adv = attack.generate(x=origin_x)
    y_test_adv = origin_y
    return x_test_adv, y_test_adv


def check_acc(model, samples, label):
    predictions = np.argmax(model.predict(samples), axis=1)
    wrong_pred_id = np.where(predictions != label)[0]
    accuracy = (len(label) - len(wrong_pred_id)) / len(label)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
    return accuracy