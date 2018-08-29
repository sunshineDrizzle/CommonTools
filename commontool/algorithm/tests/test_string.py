from commontool.algorithm.string import get_strings_by_filling


def test_get_strings_by_filling():
    string = 'a{}b{}_{}'
    fillers_list = [['0', '1'], ['0', '1'], ['YES', 'GOOD']]
    print(get_strings_by_filling(string, fillers_list))

    string = 'a{0}b{1}_{0}'
    fillers_list = [['0', '1'], ['YES', 'GOOD']]
    print(get_strings_by_filling(string, fillers_list))
