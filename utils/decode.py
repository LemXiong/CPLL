def decode(tokens, ent2id, encoding='bio'):
    id2ent = {}
    entities = {}

    for key in ent2id:
        id2ent.update({ent2id[key]: key})
        if '-' in key:
            entity_type = key.split('-')[1]
            if entity_type not in entities:
                entities.update({entity_type: []})

    index = 0
    if encoding == 'bio':
        while index < len(tokens):
            entity = id2ent[tokens[index]]
            if entity.startswith('B'):
                entity_type = entity.split('-')[1]
                start_index = index
                end_index = index
                index += 1
                while index < len(tokens):
                    current_entity = id2ent[tokens[index]]
                    if current_entity.startswith('B'):
                        break
                    elif current_entity.startswith('O'):
                        index += 1
                    elif current_entity.startswith('I'):
                        current_entity_type = current_entity.split('-')[1]
                        index += 1
                        if entity_type == current_entity_type:
                            end_index += 1
                        else:
                            break
                entities[entity_type].append((start_index, end_index))
            else:
                index += 1
    elif encoding == 'bioes':
        while index < len(tokens):
            entity = id2ent[tokens[index]]
            if entity.startswith('S'):
                entities[entity.split('-')[1]].append((index, index))
                index += 1
            elif entity.startswith('B'):
                entity_type = entity.split('-')[1]
                start_index = index
                index += 1
                while index < len(tokens):
                    current_entity = id2ent[tokens[index]]
                    if current_entity.startswith('I'):
                        index += 1
                        if entity_type != current_entity.split('-')[1]:
                            break
                    elif current_entity.startswith('E'):
                        end_index = index
                        index += 1
                        if entity_type == current_entity.split('-')[1]:
                            entities[entity_type].append((start_index, end_index))
                        else:
                            break
                    else:
                        break
            else:
                index += 1
    return entities
