list1=[10,20,30,40,50,'geeks']
list2=[10,20,'geeks']
if(set(list1).intersection(set(list2))==set(list2)):
 print("Sublist exist")
else:
 print("Sublist not exist")
l1=list(input(""))
print("List1:",l1)
l2=list(input(""))
print("List2:",l2)
flag=False
for i in range(len(l1)-len(l2)+1):
   if l1[i:i+len(l2)]==l2:
      flag=True
      break
print("Is sublist present in list....",flag)
list1=[10,20,30,40,50]
list2=[10,20]
list=False
for i in range(0,len(list1)):
 j=0
 while((i+j)<len(list1) and j<len(list2) and list1[i+j] == list2[j]):
  j+=1
 if j==len(list2):
    list=True
    break
if(list):
 print("Sublist exist")
else:
 print("Sublist not exist")
list1=[10,20,30,40,50]
list2=[10,20]
print(set(list2).issubset(set(list1)))
