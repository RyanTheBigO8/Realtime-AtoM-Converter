import sys
import csv
import numpy as np
import librosa

from ANSI_colors import Color

remove_first = True
 
def get_minimum_penalty(x, y, pxy:int, pgap:int):
    """
    Function to find out the minimum penalty
 
    :param x: pattern X
    :param y: pattern Y
    :param pxy: penalty of mis-matching the characters of X and Y
    :param pgap: penalty of a gap between pattern elements
    """
 
    # initializing variables
    i = 0
    j = 0
    mismatches = 0
    gaps = 0
     
    # pattern lengths
    m = len(x)
    n = len(y)
     
    # table for storing optimal substructure answers
    dp = np.zeros([m+1,n+1], dtype=int) #int dp[m+1][n+1] = {0};
 
    # initialising the table
    dp[0:(m+1),0] = [ i * pgap for i in range(m+1)]
    dp[0,0:(n+1)] = [ i * pgap for i in range(n+1)]
 
    # calculating the minimum penalty
    i = 1
    while i <= m:
        j = 1
        while j <= n:
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1] + pxy,
                                dp[i - 1][j] + pgap,
                                dp[i][j - 1] + pgap)
            j += 1
        i += 1

    # Reconstructing the solution
    l = n + m   # maximum possible length
    i = m
    j = n
     
    xpos = l
    ypos = l
 
    # Final answers for the respective strings
    xans = np.full(l+1, "--", dtype='U2')
    yans = np.full(l+1, "--", dtype='U2')
 
    while not (i == 0 or j == 0):
        #print(f"i: {i}, j: {j}")
        if x[i - 1] == y[j - 1]:        
            xans[xpos] = x[i - 1]
            yans[ypos] = y[j - 1]
            xpos -= 1
            ypos -= 1
            i -= 1
            j -= 1
        elif (dp[i - 1][j - 1] + pxy) == dp[i][j]:
         
            xans[xpos] = x[i - 1]
            yans[ypos] = y[j - 1]
            xpos -= 1
            ypos -= 1
            i -= 1
            j -= 1
            mismatches += 1
        elif (dp[i - 1][j] + pgap) == dp[i][j]:
            xans[xpos] = x[i - 1]
            yans[ypos] = "--"
            xpos -= 1
            ypos -= 1
            i -= 1
            gaps += 1
        elif (dp[i][j - 1] + pgap) == dp[i][j]:        
            xans[xpos] = "--"
            yans[ypos] = y[j - 1]
            xpos -= 1
            ypos -= 1
            j -= 1
            gaps += 1
 
    while xpos > 0:
        if i > 0:
            i -= 1
            xans[xpos] = x[i]
            xpos -= 1
        else:
            xans[xpos] = "--"
            xpos -= 1
     
    while ypos > 0:
        if j > 0:
            j -= 1
            yans[ypos] = y[j]
            ypos -= 1
        else:
            yans[ypos] = "--"
            ypos -= 1

 
    # Since we have assumed the answer to be n+m long,
    # we need to remove the extra gaps in the starting
    # id represents the index from which the arrays
    # xans, yans are useful
    id = 1
    i = l
    while i >= 1:
        if (yans[i] == '--') and xans[i] == '--':
            id = i + 1
            break
         
        i -= 1
 
    # Printing the final answer
    print(Color.YELLOW + "|| Aligned Sequences:" + Color.END)
    print() 

    # X
    i = id
    x_seq = np.array([], dtype='U3')
    while i <= l:
        x_seq = np.append(x_seq, xans[i])
        i += 1
    for j in range(len(x_seq)):
        if x_seq[j] != "--":
            x_seq[j] = librosa.midi_to_note(int(x_seq[j]))
        
    print(f"Aligned ANS: {x_seq}")
    print()
 
    # Y
    i = id
    y_seq = np.array([], dtype='U3')
    while i <= l:
        y_seq = np.append(y_seq, yans[i])
        i += 1
    for j in range(len(y_seq)):
        if y_seq[j] != "--":
            y_seq[j] = librosa.midi_to_note(int(y_seq[j]))
    print(f"Aligned TEST: {y_seq}")
    print()

    # print the number of mismatches and gaps
    print(Color.RED + f"Mismatches = {mismatches}" + Color.END)
    print(Color.RED + f"Gaps = {gaps}" + Color.END)
    penalty = dp[m][n]
    print(Color.RED + f"Minimum Penalty = {penalty}" + Color.END)

    # calculate score
    score = (1 - penalty / m)
    print(Color.CYAN + f"Score: {score:.3f}" + Color.END)

 

def test_get_minimum_penalty():
    """
    Test the get_minimum_penalty function
    """
    # input strings
    gene1 = np.array(["A1", "G1", "G1", "C1", "A1"], dtype=str)
    gene2 = np.array(["A1", "G1", "C1", "G1"], dtype=str)
     
    # initialising penalties of different types
    mismatch_penalty = 2
    gap_penalty = 1
 
    # calling the function to calculate the result
    get_minimum_penalty(gene1, gene2, mismatch_penalty, gap_penalty)


# test_get_minimum_penalty()

def get_NoteOn_seq():
    """
    Function to get the NoteOn sequence
    """

    # create numpy array to store the NoteOn sequence
    NoteOn_ans = np.array([], dtype=str)
    NoteOn_test = np.array([], dtype=str)

    print(Color.YELLOW + "|| Input Sequences:" + Color.END)
    print()

    ans_filename = sys.argv[1]
    with open(ans_filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if (len(row) > 6):
                if (row[6][:2] == "90"): # NoteOn
                    NoteOn_ans = np.append(NoteOn_ans, row[3][8:10])
    print(f"Input ANS: {librosa.midi_to_note(NoteOn_ans.astype(int))}")
    print()         
    
    test_filename = sys.argv[2]
    NoteOn_test = np.load(test_filename)

    # remove the first element if needed
    if remove_first:
        NoteOn_test = NoteOn_test[2:]

    print(f"Input TEST: {librosa.midi_to_note(NoteOn_test.astype(int))}")
    print()

    return NoteOn_ans, NoteOn_test
    

def main():
    """
    Main function
    """
    # get the NoteOn sequences
    NoteOn_ans, NoteOn_test = get_NoteOn_seq()

    # initialising penalties of different types
    mismatch_penalty = 1
    gap_penalty = 1

    # perform alignment and get score
    get_minimum_penalty(NoteOn_ans, NoteOn_test, mismatch_penalty, gap_penalty)
    

main()