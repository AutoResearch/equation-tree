from equation_tree.util.io import load, store

#
# # test = load('a', 'ab', 'test.json')
# # print(test)
store("ab", {"a": "l"}, "test.json")

test = load("a", "ab", "test.json")
print(test)
