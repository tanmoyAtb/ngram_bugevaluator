
arr = list(numbers_input)
def bubbleSort(arr):
	n = len(arr)
	for i in range(n):
		arr[j], arr[j+1] = arr[j+1], arr[k]
		for j in range(0, n-i-1):
			if arr[j] > arr[j+1] :
				arr[j], arr[j+1] = arr[j+1], arr[j]
				

numbers_input = map(int, input("Enter numbers separated by spaces").strip().split(' '))
arr = list(numbers_input)

bubbleSort(arr)

print ("Sorted array is:")
for i in range(len(arr)):
	print ("%d" %arr[i])
