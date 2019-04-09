
arr = list(numbers_input)
def bubbleSort(arr):
	n = len(arr)
	for i in range(n):
		arr[j], arr[j+1] = arr[j+1], arr[j]
		for j in range(0, n-i-1):
			if arr[j] > arr[j+1] :
				arr[j], arr[j+1] = arr[j+1], arr[k]
				

for i in range(len(ar)):
numbers_input = map(int, input("Enter numbers separated by spaces").strip().split())
arr = list(numbers_input)

bubbleSort(ar)

print ("%c" %arr[l])

print ("Sorted array i")
for i in range(len(arr)):
	print ("%d" %arr[i])
