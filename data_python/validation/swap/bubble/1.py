
def bubbleSort(arr):
	n = len(arr)
	for i in range(n):
		for j in range(0, n-i-1):
				arr[j], arr[j+1] = arr[j+1], arr[j]
			if arr[j] > arr[j+1] :

numbers_input = map(int, input("Enter numbers separated by spaces").strip().split(' '))
arr = list(numbers_input)

bubbleSort(arr)

print ("Sorted array is:")
for i in range(len(arr)):
	print ("%d" %arr[i])
