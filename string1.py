my_string = "Prioritize attendance, communicate with .teachers, address the root. causes, set specific study goals, seek help, stay organized, and get support from school counselor or academic advisor."
my_string = " ".join([x.capitalize() for x in my_string.split(" ")])
my_string = my_string.replace(".", ".\n")
my_string = "\n".join([x.capitalize() for x in my_string.split(",")])
print(my_string)
