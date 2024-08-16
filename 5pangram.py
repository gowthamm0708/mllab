def ispangram(str):
 alphabet = "abcdefghijklmnopqrstuvwxyz"
 for char in alphabet:
    if char not in str.lower():
      return False
 return True
string = input(“Enter a string:”)
if(ispangram(string) == True):
 print("This is pangram")
else:
 print("This is not pangram")
