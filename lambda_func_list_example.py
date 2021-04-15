import numpy as np

data = np.array([[0, 4, 32, 4,    1, 0, 10.61, 1],
                 [0, 2, 8,  4,    0, 0, 7.54,  0],
                 [0, 3, 11, 3.77, 0, 0, 4.38,  1],
                 [0, 4, 30, 1.72, 0, 0, 11.44, 1],
                 [0, 1, 17, 3.62, 1, 0, 3.06,  1],
                 [1, 3, 21, 4,    0, 1, 0.62,  1],
                 [1, 2, 30, 1.32, 0, 1, 4.3,   0],
                 [1, 3, 10, 2.72, 1, 0, 4.89,  0]])

features = data[:, :-1]
classes = data[:, -1]

print(features)
print(classes)

func_list = [lambda features, avg, i: features[i] > avg,
             lambda features, med, i: features[i] >= med]
val_list = [np.average(data, axis=0),
            np.median(data, axis=0)]
fv_list = zip(func_list, val_list)

for func, val in fv_list:
    print("----working in {} with val {}---".format(func, val))
    for i in range(len(features[0])):
        print("--Currently on attribute {}".format(i))
        left_set = []
        right_set = []
        for record in features:
            if func(record, val[i], i):
                left_set.append(record)
            else:
                right_set.append(record)
        print("----left children are---")
        for c in left_set:
            print(c)
        print("----right children are---")
        for c in right_set:
            print(c)


