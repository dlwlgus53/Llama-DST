original_list = [
    {"id": 1, "name": "John"},
    {"id": 2, "name": "Alice"},
    {"id": 1, "name": "John"},
    {"id": 3, "name": "Bob"},
    {"id": 2, "name": "Alice"}
]

# Create a new list to store unique dictionaries
unique_list = []

for d in original_list:
    if d not in unique_list:
        unique_list.append(d)

print(unique_list)
