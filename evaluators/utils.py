def multiple_index_from_attribute_list(attribute_list, indices):
    attributes = []
    for idx in indices:
        attributes.append(attribute_list[idx.item()])
    return attributes
