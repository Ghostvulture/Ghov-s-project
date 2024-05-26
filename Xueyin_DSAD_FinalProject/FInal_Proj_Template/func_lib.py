# funcition_lib.py (need to finish)

# A basic function libary which is suitable for a large search project construction and block-testing

#---------------------------- Split line ----------------------------#

# Please write your code answer below, and run the code to check whether the outputs of your method are correct.

# Note: We strongly suggest you to follow the given code template to help you think and organize your code structure.
#       Still, any changes are supported with corresponding clear comments.


# You can choose whether to use the demo codes below by yourself.

# If you modify the code template, please make sure that the corresponding testing usages will be added in your programm.
# Otherwise, we think that your answer is not valid (without promising outputs)

#---------------------------- Split line ----------------------------#
# Targets:

# Implement a prefix tree (Trie) for efficient storage and retrieval of words along with functions to perform spell checking and process queries with logical operations like AND (`+`) and OR (`|`).
# It includes functionality to find exact matches and approximate matches based on edit distance with a threshold, leveraging dynamic programming within the Trie structure.

#---------------------------- Split line ----------------------------#

# Note 1: The expressions can include operands that are effectively sets of results from previous searches (either exact or approximate word matches), combined using logical operators.



from collections import defaultdict
from math import ceil
from functools import cache
from hyperpara import *
from re import split


# Trie-based search algorithm

# METHODS:
    # inserting words into the trie
    # finding exact matches
    # finding approximate matches (with a given error threshold)

# Hint 1: Cache for speeding up future queries. This will help to quickly provide results without recalculating.
# Hint 2: Considering sometimes the inputs from users are just some substrings of one complete word, or some words which are not all exactly correct (misspelled),


class TrieNode:

    # `TrieNode`:
    #             Represents a node in the trie.
    #             Each node has a dictionary of child nodes (`children`) indexed by characters,
    #                           a flag indicating whether it marks the end of a word (`is_end_of_word`)
    #                       and a list of indices where the word appears in the original dataset (`index_of_words`).

    def __init__(self):
        self.children = {}
        self.is_end_of_word = False #用于计算误差
        self.index_of_words = []    #用于输出包含该词的行，代表行数

class Trie:

    # `Trie`:
    #         The trie class containing the root node and a cache for misspelled words.

    def __init__(self):
        self.root = TrieNode()
         # Initialize a cache for misspelled words
        self.mispelled_cache = {}
        self.lines = [] #store lines for output

    #insert word into trie tree
    def insert(self, word, index_of_line):
        # TODO
        currentNode = self.root
        for c in word:
            #insert each letter
            if c not in currentNode.children:
                newNode = TrieNode()
                currentNode.children[c] = newNode
            #insert the next letter to currentNode.children
            currentNode = currentNode.children[c]
        currentNode.is_end_of_word = True
        currentNode.index_of_words.append(index_of_line)
        
        pass

    @cache  #自动缓存findCandidates的结果, not used
    def findCandidates(self, word):
        # Note first to check if the word is in the cache
        if word in self.mispelled_cache:
            return self.mispelled_cache[word]
        
        # TODO
        exact_match = self.find_exact(word)
        if exact_match:
            self.mispelled_cache[word] = exact_match
            return exact_match
        
        approximate_match = self.find_approximate(word)
        self.mispelled_cache[word] = approximate_match
        return approximate_match
        pass

        # return results

    #第一种情况，len(word) small,directly find keyword in substring 
    def find_exact(self, word):
        # TODO
        result = {}
        result[0] = []
        for i in range(len(self.lines)):
            line = self.lines[i]
            words = split(r'\W+', line)
            for w in words:
                if word in w.lower():
                    result[0].append(i)
                    break
        #return a dict:{'error=0' : [list including index_of_lines]}
        return result
    
    #此处调用_search
    #find_approximate: find keyword in substring by trie nodes
    def find_approximate(self, word, dist=ERROR_THRESHOLD):
        # TODO

        #function defined to clean the result dict
        def deduplicate_results(results):
            deduplicated = {}
            seen = set()
            for error in sorted(results.keys()):
                deduplicated[error] = []
                for line_number in results[error]:
                    if line_number not in seen:
                        deduplicated[error].append(line_number)
                        seen.add(line_number)
            return deduplicated

        #ERROR_THRESHOLD handling
        if dist >= 1:
            dist = dist
        else:
            dist = ERROR_THRESHOLD * len(word)

        #initialize results dict
        results = {}    #key:error; value: a list of line_number at this error_count

        #first find exact: error=0
        result_exact = self.find_exact(word)
        if result_exact is not None:
            results[0] = result_exact[0]

        #then find approximate by _search function
        self._search(self.root, word, [i for i in range(len(word) + 1)], dist, results)
        #clean results
        results = deduplicate_results(results)
        return results
        pass

    #用于计算误差范围内的词
    def _search(self, node, word, previous_row, dist, results, word_so_far = ''):

        #edit_distance: use dynamic programming for substring and keyword comparison
        def edit_distance(substring, word):
            m, n = len(substring), len(word)
            dp = [[0] * (n+1) for _ in range(m+1)]
            for i in range(m+1):
                dp[i][0] = i
            for j in range(n+1):
                dp[0][j] = j

            for i in range(1, m+1):
                for j in range(1, n+1):
                    if substring[i-1] == word[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+1)

            return dp[m][n]

        #calculate error for each substring
        for length in range(len(word) - int(dist), len(word) + int(dist)):
                if len(word_so_far) >= length:
                    for start in range(len(word_so_far) - length + 1):
                        substring = word_so_far[start:start + length]
                        error = edit_distance(substring, word)
                        if error <= dist and node.is_end_of_word:
                            if error not in results:
                                results[error] = []
                            results[error].extend(node.index_of_words)

        #go further on the trie node
        for char in node.children:
            self._search(node.children[char], word, previous_row, dist, results, word_so_far + char)

        pass

#---------------------------- Split line ----------------------------#



#---------------------------- Split line ----------------------------#

# Bouns_part

def op_AND(a, b):
    # TODO
    pass

def op_OR(a, b):
    # TODO
    pass


# Given codes to help to parse and evaluate the expressions (not need to implement, but please read it carefully)

def precedence(op):
    """Return the precedence of the given operator."""
    if op == '+':
        return 2
    elif op == '|':
        return 1
    return 0

def tokenize(expression):
    """Convert the string expression into a list of tokens with implicit '+'."""
    tokens = []
    i = 0
    last_char = None

    while i < len(expression):
        if expression[i].isspace():
            i += 1
        elif expression[i].isalnum():  # Operand
            start = i
            while i < len(expression) and expression[i].isalnum():
                i += 1
            token = expression[start:i]

            # If last token is also an operand, insert an implicit '+'
            if tokens and tokens[-1].isalnum():
                tokens.append('+')
            tokens.append(token)
        else:  # Operator or parenthesis
            tokens.append(expression[i])
            i += 1

    return tokens

def infix_to_postfix(tokens):
    """Convert infix expression to postfix using the Shunting Yard algorithm."""
    stack = []
    output = []

    for token in tokens:
        if isinstance(token, dict):  # Operand
            output.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # pop '('
        else:  # Operator
            while stack and precedence(stack[-1]) >= precedence(token):
                output.append(stack.pop())
            stack.append(token)

    while stack:
        output.append(stack.pop())

    return output

# Note 2: The `evaluate_postfix` function processes a postfix expression which simplifies the evaluation of expressions by eliminating the need for parentheses and making operator processing straightforward.

def evaluate_postfix(tokens):
    """Evaluate a postfix expression"""
    stack = []
    
    tokens.append(defaultdict(lambda: float('inf')))
    tokens.append('|')
    
    for token in tokens:
        if token == '+':
            b = stack.pop()
            a = stack.pop()
            result = op_AND(a, b)
            stack.append(result)
        elif token == '|':
            b = stack.pop()
            a = stack.pop()
            result = op_OR(a, b)
            stack.append(result)
        else:  # Operand, Set
            stack.append(token)  # Convert '0' or '1' to False or True
    
    return stack.pop()
