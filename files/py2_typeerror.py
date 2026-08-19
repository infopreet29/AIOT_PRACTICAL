def total_marks(marks_list):
    total = "Total: " + sum(marks_list)   # can't concatenate str + int
    return total

marks = [80, 90, 75, 60]
print(total_marks(marks))
