def longest_same_seq_length(seq):

    if not seq:
        return 0

    sub_seq_idx = 0
    max_count = 0
    max_idx = len(seq) - 1
    while True:
        count = 0
        for idx, val in enumerate(seq[sub_seq_idx:], sub_seq_idx):
            if seq[sub_seq_idx] == val:
                count += 1
            else:
                if count > max_count:
                    max_count = count
                sub_seq_idx = idx
                break
        if idx == max_idx and sub_seq_idx < max_idx:
            if count > max_count:
                max_count = count
            break
        elif sub_seq_idx == max_idx:
            if max_count < 1:
                max_count = 1
            break
    return max_count

if __name__ == "__main__":
    print(longest_same_seq_length("122333444455555"))
    print(longest_same_seq_length(['a', 'b', 'b', 'c']))
    print(longest_same_seq_length([]))
    print(longest_same_seq_length((1,)))
