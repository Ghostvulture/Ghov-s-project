# data_handling.py (need to finish)

# Here in your project, it works for handling the input file and search for the result string by using the trie-based search algorithm (partially implemented in the `func_lib`)

#---------------------------- Split line ----------------------------#

# Please write your code answer below, and run the code to check whether the outputs of your method are correct.

# Note: We strongly suggest you to follow the given code template to help you think and organize your code structure.
#       Still, any changes are supported with corresponding clear comments.


# You can choose whether to use the demo codes below by yourself.

# If you modify the code template, please make sure that the corresponding testing usages will be added in your programm.
# Otherwise, we think that your answer is not valid (without promising outputs)

#---------------------------- Split line ----------------------------#


# Targets:
    # Implement the function load_words_from_file()
    # Implement the function search() to complete the search algorithm
    # Use the unittest to check your codes


from func_lib import Trie, tokenize, infix_to_postfix, evaluate_postfix     # You choose the modules you want
from re import split
from hyperpara import TEST_DATA_01, TEST_DATA_02, APPROX_LEN_THRESHOLD, ERROR_THRESHOLD
import unittest


def load_words_from_file(file_path):

    # TODO
    trie = Trie()
    # read a text file
    with open(file_path, 'r') as file:
        for line_index, line in enumerate(file):
            #add line into the tree for output
            trie.lines.append(line)
            # process its content by splitting it into words based on non-alphanumeric characters
            words = split(r'\W+', line)
            # then insert these words into a Trie data structure along with their positions in the file.
            for word in words:
                if word.isalpha():
                    trie.insert(word.lower(), line_index)

    return trie
    pass


def search(keywords, trie):

    # TODO
    keyword_list = keywords.split()
    keyword_res = []

    # processes a keyword-based search query
    for keyword in keyword_list:
        if len(keyword) < APPROX_LEN_THRESHOLD:
            keyword_res.append(trie.find_exact(keyword.lower()))
        else:
            approximate_matches = trie.find_approximate(keyword.lower(),dist=ERROR_THRESHOLD)
            keyword_res.append(approximate_matches)

    #when dealing with multiple words, do the intersection
    if not keyword_res:
        return {}
    common_res = keyword_res[0]

    for kw_res in keyword_res[1:]:
        new_common_res = {}
        for error1,lines1 in common_res.items():
            for error2, lines2 in kw_res.items():
                common_lines = set(lines1).intersection(set(lines2))
                if common_lines:
                    common_error = error1 + error2
                    if common_error not in new_common_res:
                        new_common_res[common_error] = []
                    new_common_res[common_error].extend(common_lines)
        common_res = new_common_res

    #sort results by errors and index
    sorted_res = sorted(common_res.items(), key = lambda x : x[0])
    top_20_res = sorted_res[:20]


    # returns the top 20 matching records as a formatted string.
    # return "Output:\n" + "\n".join(ss)
    output_lines = []
    for error, line_numbers in top_20_res:
        line_numbers.sort()
        for line_num in line_numbers:
            output_lines.append(f"{trie.lines[line_num].strip()}")#remove additional '\n'


    return "Output:\n" + "\n".join(output_lines)


    pass
    



###################### Test 1 ######################

# unit testing 01

# APPROX_LEN_THRESHOLD = 3
# ERROR_THRESHOLD = 1

# class TestSearchFunction(unittest.TestCase):
#     def setUp(self):
#         # Set up a trie and load records for testing
#         self.trie = load_words_from_file(TEST_DATA_01)
#         self.maxDiff = None

#     def test_exact_matching(self):
#         # Test with keywords that do not match any sentence
#         result = search("gy", self.trie)
#         expected_result = '''Output:\n2,A Data-Driven Method to Detect the Abnormal Instances in an Electricity Market,"electricity market,data mining,anomaly detection",Machine Learning in Energy Applications,2015\n4,A knowledge growth and consolidation framework for lifelong machine learning systems,"lifelong machine learning, oblivion criterion, knowledge topology and acquisition, declarative learning",Machine Learning I,2014\n5,A Hybrid Genetic-Programming Swarm-Optimisation Approach for Examining the Nature and Stability of High Frequency Trading Strategies,"sociology, statistics, noise, testing, prediction algorithms, algorithm design and analysis, genetics",Real-time Systems and Industry,2014'''
#         #print(expected_result)
#         self.assertEqual(result, expected_result)

#     def test_appr_matching(self):
#         # Test with keywords that do not match any sentence
#         print(APPROX_LEN_THRESHOLD, ERROR_THRESHOLD)
#         result = search("ogy", self.trie)
#         expected_result = '''Output:\n4,A knowledge growth and consolidation framework for lifelong machine learning systems,"lifelong machine learning, oblivion criterion, knowledge topology and acquisition, declarative learning",Machine Learning I,2014\n5,A Hybrid Genetic-Programming Swarm-Optimisation Approach for Examining the Nature and Stability of High Frequency Trading Strategies,"sociology, statistics, noise, testing, prediction algorithms, algorithm design and analysis, genetics",Real-time Systems and Industry,2014\n1,Prediction of Sunspot Number Using Minimum Error Entropy Cost Based Kernel Adaptive Filters,"kernel methods,error entropy,information theoretic learning",Machine Learning Algorithms for Environmental Applications ,2015\n2,A Data-Driven Method to Detect the Abnormal Instances in an Electricity Market,"electricity market,data mining,anomaly detection",Machine Learning in Energy Applications,2015'''
#         self.assertEqual(result, expected_result)

#     def test_appr_matching(self):
#         # Test with keywords that do not match any sentence
#         result = search("logy", self.trie)
#         expected_result = '''Output:\n4,A knowledge growth and consolidation framework for lifelong machine learning systems,"lifelong machine learning, oblivion criterion, knowledge topology and acquisition, declarative learning",Machine Learning I,2014\n5,A Hybrid Genetic-Programming Swarm-Optimisation Approach for Examining the Nature and Stability of High Frequency Trading Strategies,"sociology, statistics, noise, testing, prediction algorithms, algorithm design and analysis, genetics",Real-time Systems and Industry,2014'''
#         self.assertEqual(result, expected_result)

###################### Test 1 ######################


###################### Test 2 ######################

# unit testing 02

# APPROX_LEN_THRESHOLD = 5
# ERROR_THRESHOLD = 2

# class TestSearchFunction(unittest.TestCase):
#     def setUp(self):
#         # Set up a trie and load records for testing
#         self.trie = load_words_from_file(TEST_DATA_02)
#         self.maxDiff = None



#     def test_exact_plus_app_matching(self):
#         # Test with keywords that do not match any sentence
#         result = search("conv Learn", self.trie)
#         expected_result = '''Output:\n1,Learning Good Features To Track,"object tracking, convolutional neural network, feature learning",Feature Extraction and Selection,2014\n3,A Cyclic Contrastive Divergence Learning Algorithm for High-order RBMs,"high-order rbms, cyclic contrastive divergence learning, gradient approximation, convergence, upper bound",Neural Networks I,2014\n2,Human action recognition based on recognition of linear patterns in action bank features using convolutional neural networks,"human action recognition, action bank features, deep convolutional network",Neural Networks I,2014'''
#         self.assertEqual(result, expected_result)

#     def test_misspell_mix_matching(self):
#         # Test with keywords that do not match any sentence
#         result = search("Fearu Netwo", self.trie)
#         expected_result = '''Output:\n1,Learning Good Features To Track,"object tracking, convolutional neural network, feature learning",Feature Extraction and Selection,2014\n2,Human action recognition based on recognition of linear patterns in action bank features using convolutional neural networks,"human action recognition, action bank features, deep convolutional network",Neural Networks I,2014\n9,Multi-Variable Neural Network Forecasting Using Two Stage Feature Selection,"forecasting, feature selection, neural networks",Neural Network II,2014\n3,A Cyclic Contrastive Divergence Learning Algorithm for High-order RBMs,"high-order rbms, cyclic contrastive divergence learning, gradient approximation, convergence, upper bound",Neural Networks I,2014\n5,Improving Performance on Problems with Few Labelled Data by Reusing Stacked Auto-Encoders,"transfer learning, deep learning, artificial neural networks",Neural Networks I,2014\n10,Adaptive restructuring of radial basis functions using integrate-and-fire neurons,"machine learning, radial basis functions, neural networks, feed-forward networks",Neural Network II,2014'''
#         self.assertEqual(result, expected_result)

###################### Test 2 ######################


###################### Test 3 ######################

# unit testing 03

# APPROX_LEN_THRESHOLD = 5
# ERROR_THRESHOLD = 1

# class TestSearchFunction(unittest.TestCase):
#     def setUp(self):
#         # Set up a trie and load records for testing
#         self.trie = load_words_from_file(TEST_DATA_02)    # You are allowed to change returned variables here. Still, you need to change correspondingly the unit test by yourself.
#         self.maxDiff = None

#     def test_and_or_mix_matching(self):
#         # Test with keywords that do not match any sentence
#         result = search("netwo (conv | activ)", self.records, self.root)  # You are allowed to change returned variables here. Still, you need to change correspondingly the unit test by yourself.
#         expected_result = '''Output:\n1,Learning Good Features To Track,"object tracking, convolutional neural network, feature learning",Feature Extraction and Selection,2014\n2,Human action recognition based on recognition of linear patterns in action bank features using convolutional neural networks,"human action recognition, action bank features, deep convolutional network",Neural Networks I,2014\n3,A Cyclic Contrastive Divergence Learning Algorithm for High-order RBMs,"high-order rbms, cyclic contrastive divergence learning, gradient approximation, convergence, upper bound",Neural Networks I,2014\n4,Facial expression recognition using kinect depth sensor and convolutional neural networks,"convolutional neural networks (cnn), facial expression recognition",Neural Networks I,2014\n7,Human action recognition based on MOCAP information using convolution neural networks,"convolutional neural networks (cnn), motion capture (mocap)",Neural Network II,2014\n11,One-shot periodic activity recognition using Convolutional Neural Networks,"human activity recognition, convolutional neural networks (cnn)",Neural Network II,2014\n10,Adaptive restructuring of radial basis functions using integrate-and-fire neurons,"machine learning, radial basis functions, neural networks, feed-forward networks",Neural Network II,2014'''
#         self.assertEqual(result, expected_result)

###################### Test 3 ######################

# Run the tests
if __name__ == "__main__":
    unittest.main()