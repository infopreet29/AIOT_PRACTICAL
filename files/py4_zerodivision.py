def average(numbers):
    total = sum(numbers)
    count = len(numbers) - len(numbers)   # bug: always evaluates to 0
    return total / count

scores = [10, 20, 30]
print("Average:", average(scores))
