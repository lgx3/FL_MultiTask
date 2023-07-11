import numpy as np
from options import args_parser



## 20个客户端 0 - 0.9 各2个
def dfs(x, summ, target_left, target_right, lef, q_num, a, num):
    if summ > lef:
        return 0
    if x == num - 1:
        a[num - 1] = lef - summ
        a_sum = np.sum(np.square(a - q_num / 10))
        if a_sum <= target_right and a_sum >= target_left:
            for j in range(num):
                category_nums[j] = a[j]
            return 1
        return 0
    for i in range(30, lef - summ, 60):
        a[x] = i
        if dfs(x + 1, summ + a[x], target_left, target_right, lef, q_num, a, num):
            return 1
    return 0

q_num = 400
gen_catnum_base_emd = {0.9 : {}}
arr = np.zeros(10)
arr[0] += q_num
gen_catnum_base_emd[0.9][1] = arr
for i in range(1, 9):
    emd = 0.1 * (9 - i)
    gen_catnum_base_emd[emd] = {}
    for j in range(1, 11):
        target_left = pow(emd * q_num, 2) - (10 - j) * pow((q_num / 10), 2)
        target_right = pow((emd + 0.099) * q_num, 2) - (10 - j) * pow((q_num / 10), 2)
        category_nums = np.zeros(10)
        if dfs(0, 0, target_left, target_right, q_num, q_num, np.zeros(j, 'int'), j) == 1:
            gen_catnum_base_emd[emd][j] = category_nums
gen_catnum_base_emd[0] = {}
gen_catnum_base_emd[0][10] = np.array([q_num / 10] * 10)



## emd 0 - 0.9 2个client一个 ，标签类别随机
def get_every_clients_category_nums3(gen_catnum_base_emd):
    args = args_parser()
    client = np.arange(0, args.num_users)
    np.random.shuffle(client) ## 随机
    category_nums = np.zeros((args.num_users, 10))
    clas= np.zeros(args.num_users)
    for i in range(0, 10):
        emd = i * 0.1
        for j in range(2):
            ## 随机生成类别数量，如果存在的话
            num_of_cato = 0
            if emd == 0:
                num_of_cato = 10
            elif emd == 0.9:
                num_of_cato = 1
            else:
                while(True):
                    num_of_cato = np.random.randint(1, 11)
                    if num_of_cato in gen_catnum_base_emd[emd]:
                        break
            np.random.shuffle(gen_catnum_base_emd[emd][num_of_cato])
            category_nums[client[i * 2 + j]] = gen_catnum_base_emd[emd][num_of_cato]
            clas[client[i * 2 + j]] = num_of_cato
    return category_nums, clas


#获得每个类别的索引
def get_every_category_index(y_data):
    indd = []
    for k in range(10):
        ind = []
        for i in range(y_data.shape[0]):
            if (int)(y_data[i]) == k:
                ind.append(i)
        indd.append(ind)
    return indd



def emd_data_divide(dataset_train):
    args = args_parser()
    dict_users = {i: np.array([], dtype='int64') for i in range(args.num_users)}
    labels = dataset_train.targets.numpy()
    ind_for_category = get_every_category_index(labels)
    q = np.zeros(args.num_users)
    for i in range(args.num_users):
        q[i] = 60000 / args.num_users
    category_nums, clas = get_every_clients_category_nums3(gen_catnum_base_emd)  # 每个客户端类别的数量
    # print(category_nums)

    return category_nums, clas