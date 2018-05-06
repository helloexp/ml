
def main():
    arr = [1,2,4,100]
    res = solve(arr)
    print(res)


def solve(arr):
    length = len(arr)
    max_arr = [-1] * length
    max_arr[0] = arr[0]
    max_arr[1] = arr[1]
    max_arr[2] = arr[0] + arr[2]

    for i in range(3, length):
        max_arr[i] = max(arr[i] + max_arr[i - 2], arr[i] + max_arr[i - 3])

    return max(max_arr[length - 1], max_arr[length - 2])


if __name__ == '__main__':
    main()
