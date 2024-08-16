def dict(dictionary):
 values = list(dictionary.values())
 average = sum(values) / len(values)
 new_dict = {key: average for key in dictionary}
 return new_dict
dict1 = {'a': 10, 'b': 20, 'c': 30}
new_dict= dict(dict1)
print("Dictionary:",dict1)
print("Values Replaced by Average:", new_dict)
