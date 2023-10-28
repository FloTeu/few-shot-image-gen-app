
def extract_list_items(string_with_enumerate):
    items = string_with_enumerate.split("\n")
    extracted_list = []

    for item in items:
        index = item.find(". ")
        if index != -1:
            extracted_list.append(item[index + 2:])

    return extracted_list