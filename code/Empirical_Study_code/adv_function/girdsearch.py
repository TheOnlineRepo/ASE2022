from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import os

lgb_params = {"learning_rate": 0.1,
              "feature_fraction": 0.7,
              "min_child_samples": 21,
              "min_child_weight": 0.001, }
lgb_adj_1 = {'max_depth': [4, 6, 8],
             'num_leaves': [20, 30, 40]}
lgb_adj_2 = {'min_child_samples': [18, 19, 20, 21, 22],
             'min_child_weight': [0.001, 0.002]}
lgb_adj_3 = {'feature_fraction': [0.6, 0.8, 1]}
lgb_adj_4 = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
lgb_adj_list = [lgb_adj_1]

xgb_params = {'learning_rate': 0.1,
              'n_estimators': 500,
              'max_depth': 5,
              'min_child_weight': 1,
              'seed': 0,
              'gamma': 0,
              'reg_alpha': 0,
              'reg_lambda': 1}
xgb_adj_1 = {'n_estimators': [300, 350, 400, 450, 500, 550, 600, 650, 700]}
xgb_adj_2 = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
             'min_child_weight': [1, 2, 3, 4, 5, 6]}
xgb_adj_list = [xgb_adj_1, xgb_adj_2]

svc_params = {}
svc_adj_list = [{'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}]

rf_params = {}
rf_adj_list = {"n_estimators": [10, 15, 20],
               "criterion": ["gini", "entropy"],
               "min_samples_leaf": [2, 4, 6]}


def param_adj(base_model, adj_list, base_param, train_x, train_y, test_x, test_y, log_file):
    global acc
    for i, adj_param in enumerate(adj_list):
        model = base_model(**base_param)
        opti_model = GridSearchCV(estimator=model, param_grid=adj_param, scoring='roc_auc', cv=3, verbose=0, n_jobs=4)
        opti_model.fit(train_x, train_y)

        pred_y = opti_model.predict(test_x)
        acc = accuracy_score(test_y, pred_y)

        print(f"———————— Round {i} ——————————")
        print("Test Acc:", acc)
        print('参数的最佳取值：{0}'.format(opti_model.best_params_))
        if log_file is not None:
            with open(log_file, 'a+') as f:
                f.write(f"———————— Round {i} ——————————\n")
                f.write(f"Test Acc: {acc} \n")
                f.write('参数的最佳取值：{0}\n'.format(opti_model.best_params_))

        for key in opti_model.best_params_:
            base_param[key] = opti_model.best_params_[key]
    return str(base_param), acc, opti_model


def get_save_dir(path, dataset, model, adv_method, fitness, param):
    adv_save_dir = os.path.join(path, 'adv_result')
    dataset_dir = os.path.join(adv_save_dir, dataset)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    model_dir = os.path.join(dataset_dir, model)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    adv_method_dir = os.path.join(model_dir, adv_method)
    if not os.path.exists(adv_method_dir):
        os.mkdir(adv_method_dir)
    fitness_dir = os.path.join(adv_method_dir, fitness)
    if not os.path.exists(fitness_dir):
        os.mkdir(fitness_dir)
    param_dir = os.path.join(fitness_dir, str(param))
    if not os.path.exists(param_dir):
        os.mkdir(param_dir)
    return param_dir
