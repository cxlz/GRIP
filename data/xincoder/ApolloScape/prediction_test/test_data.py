test_file = "data/xincoder/ApolloScape/prediction_test/prediction_test.txt"
test_gt_file = "data/xincoder/ApolloScape/prediction_test/prediction_gt.txt"
test_out_file = "data/xincoder/ApolloScape/prediction_test/prediction_test0.txt"
lst = []
with open(test_file) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(" ")
        lst.append(line)
with open(test_gt_file) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(" ")
        for i in range(len(line), 10):
            line.append("0")
        lst.append(line)
lst = sorted(sorted(lst, key=lambda x: int(x[1])), key=lambda x: int(x[0]))
with open(test_out_file, "w+") as f:
    for line in lst:
        f.write(" ".join(line))
        f.write("\n")