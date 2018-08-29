def _filling(string, fillers_list, fillers, strings):
    """
    do filling

    Parameters:
    -----------
    string: string
        A string that may contain several placeholders {}.
    fillers_list: list
        Each element is a collection of fillers named fillers_c. Each fillers_c works on
        its corresponding placeholder {} in the 'string'. And there is an one-to-one
        correspondence in order between fillers_c and placeholders, which can also be specified
        by position parameters. Fillers in these fillers_cs are used to fill corresponding placeholders
        to generate strings we need.
        The number of generated string is equal to
        len(fillers_list[0]) * len(fillers_list[1]) * ... * len(fillers_list[-1])
    fillers: list
        There are some fillers used to fill the 'string' at one iteration in this list.
    strings: list
        Modified in situ to store generated strings.

    """
    idx = len(fillers)
    if idx != len(fillers_list):
        for filler in fillers_list[idx]:
            _filling(string, fillers_list, fillers + [filler], strings)
    else:
        strings.append(string.format(*fillers))


def get_strings_by_filling(string, fillers_list):
    """
    get strings by filling

    Parameters:
    -----------
    string: string
        A string that may contain several placeholders {}.
    fillers_list: list
        Each element is a collection of fillers named fillers_c. Each fillers_c works on
        its corresponding placeholder {} in the 'string'. And there is an one-to-one
        correspondence in order between fillers_c and placeholders, which can also be specified
        by position parameters. Fillers in these fillers_cs are used to fill corresponding placeholders
        to generate strings we need.
        The number of generated string is equal to
        len(fillers_list[0]) * len(fillers_list[1]) * ... * len(fillers_list[-1])
    """
    strings = []
    _filling(string, fillers_list, [], strings)

    return strings
