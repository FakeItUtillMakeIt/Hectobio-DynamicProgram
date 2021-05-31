#全局匹配和局部匹配算法

import numpy as np
import sys


def theta(a, b):
    if a == '-' or b == '-' or a != b:   # gap or mismatch
        return -1
    elif a == b:                         # match
        return 1


def make_score_matrix(seq1, seq2):
    """
    return score matrix and map(each score from which direction)
    0: diagnosis
    1: up
    2: left
    """
    seq1 = '-' + seq1
    seq2 = '-' + seq2
    score_mat = {}
    trace_mat = {}

    for i, p in enumerate(seq1):
        score_mat[i] = {}
        trace_mat[i] = {}
        for j, q in enumerate(seq2):
            if i == 0:                    # first row, gap in seq1
                score_mat[i][j] = -j
                trace_mat[i][j] = 1
                continue
            if j == 0:                    # first column, gap in seq2
                score_mat[i][j] = -i
                trace_mat[i][j] = 2
                continue
            ul = score_mat[i-1][j-1] + theta(p, q)     # from up-left, mark 0
            # from left, mark 1, gap in seq1
            l = score_mat[i][j-1] + theta('-', q)
            # from up, mark 2, gap in seq2
            u = score_mat[i-1][j] + theta(p, '-')
            picked = max([ul, l, u])
            score_mat[i][j] = picked
            trace_mat[i][j] = [ul, l, u].index(
                picked)   # record which direction
    return score_mat, trace_mat


def traceback(seq1, seq2, trace_mat):
    '''
    find one optimal traceback path from trace matrix, return path code
    -!- CAUTIOUS: if multiple equally possible path exits, only return one of them -!-
    '''
    seq1, seq2 = '-' + seq1, '-' + seq2
    i, j = len(seq1) - 1, len(seq2) - 1
    path_code = ''
    while i > 0 or j > 0:
        direction = trace_mat[i][j]
        if direction == 0:                    # from up-left direction
            i = i-1
            j = j-1
            path_code = '0' + path_code
        elif direction == 1:                  # from left
            j = j-1
            path_code = '1' + path_code
        elif direction == 2:                  # from up
            i = i-1
            path_code = '2' + path_code
    return path_code


def print_m(seq1, seq2, m):
    """print score matrix or trace matrix"""
    seq1 = '-' + seq1
    seq2 = '-' + seq2
    print()
    print(' '.join(['%3s' % i for i in ' '+seq2]))
    for i, p in enumerate(seq1):
        line = [p] + [m[i][j] for j in range(len(seq2))]
        print(' '.join(['%3s' % i for i in line]))
    print()
    return


def pretty_print_align(seq1, seq2, path_code):
    '''
    return pair alignment result string from
    path code: 0 for match, 1 for gap in seq1, 2 for gap in seq2
    '''
    align1 = ''
    middle = ''
    align2 = ''
    for p in path_code:
        if p == '0':
            align1 = align1 + seq1[0]
            align2 = align2 + seq2[0]
            if seq1[0] == seq2[0]:
                middle = middle + '|'
            else:
                middle = middle + ' '
            seq1 = seq1[1:]
            seq2 = seq2[1:]
        elif p == '1':
            align1 = align1 + '-'
            align2 = align2 + seq2[0]
            middle = middle + ' '
            seq2 = seq2[1:]
        elif p == '2':
            align1 = align1 + seq1[0]
            align2 = align2 + '-'
            middle = middle + ' '
            seq1 = seq1[1:]

    print('Alignment:\n\n   ' + align1 + '\n   ' +
          middle + '\n   ' + align2 + '\n')
    return


def usage():
    print('Usage:\n\tpython nwAligner.py seq1 seq2\n')
    return


def Needleman_Wunsch():
    # 全局比对算法
    # Needleman-Wunsch比对算法
    try:
        seq1, seq2 = map(str.upper, sys.argv[1:3])
    except:
        seq1, seq2 = 'GGGAAATCCCATTGCATAGGTAAAAAAAGGTAGCTCAGCTTGGGGA', 'TTTTCCCGTCCGCTCGTTCGCGCCTGTCTGCTTGTTTGCGTTTT'
        usage()
        print('--------Demo:-------\n')

    print('1: %s' % seq1)
    print('2: %s' % seq2)

    score_mat, trace_mat = make_score_matrix(seq1, seq2)
    #print_m(seq1, seq2, score_mat)
    #print_m(seq1, seq2, trace_mat)

    path_code = traceback(seq1, seq2, trace_mat)
    pretty_print_align(seq1, seq2, path_code)
    #print('   '+path_code)


#

# smith-waterman算法
def Trace_back(str1, str2, M, Space):
    # find max
    x, y = np.where(M == np.max(M))
    #找出str1的长度和与str1匹配的概率最大的长度
    x, y = x[0], y[0]
    # print(M)
    #print(x, y)
    match_str1, match_str2 = '', ''
    match_count = 0
    score = 0
    count = 0
    #回溯
    while M[x, y] != 0:
        count += 1
        # print(x, y)
        #往上走，str1有叠词或有没匹配到的词
        if M[x - 1, y] - Space == M[x, y]:
            x = x - 1
            match_str1, match_str2 = str1[x] + match_str1, '_' + match_str2
            score += 0.5
        #往左走，str2有叠词或没匹配到的词
        elif M[x, y - 1] - Space == M[x, y]:
            y = y - 1
            match_str1, match_str2 = '_' + match_str1, str2[y] + match_str2
            score += 0.5
        #对角线方向，两个str匹配且目前均无叠词
        else:
            x, y = x-1, y-1
            match_str1, match_str2 = str1[x] + match_str1, str2[y] + match_str2
            match_count += 1
            score += 1
        # match_rate = match_count/min(len(str1), len(str2))
    return match_str1, match_str2, score/count


#局部匹配算法
def Smith_Waterman(str1, str2, s_score, m_score):
    len1, len2 = len(str1), len(str2)
    matrix = np.zeros([len1 + 1, len2 + 1])
    for i in range(len1):
        matrix[i, 0] = 0
    for i in range(len2):
        matrix[0, i] = 0
    Space = 0
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            Mkj = matrix[i-1, j] - Space
            Mik = matrix[i, j-1] - Space
            Mij = matrix[i-1, j-1] + 1 if str1[i -
                                               1] == str2[j-1] else matrix[i-1, j-1] - 1
            matrix[i, j] = max(Mij, Mkj, Mik, 0)

    match_str1, match_str2, match_rate = Trace_back(str1, str2, matrix, Space)

    # print(match_str1)
    # print(match_str2)
    # print(match_rate)
    return match_str1, match_str2, match_rate


if __name__ == '__main__':
    Needleman_Wunsch()
    str1 = 'GGGAAATCCCATTGCATAGGTAAAAAAAGGTAGCTCAGCTTGGGGA'
    str2 = 'TTTTCCCGTCCGCTCGTTCGCGCCTGTCTGCTTGTTTGCGTTTT'
    print(Smith_Waterman(str1, str2, 0.5, 1))
