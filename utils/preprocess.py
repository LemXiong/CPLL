import json
import os.path
import random
import sys
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', default='data/resume', type=str)
    parser.add_argument('--output_dir', default='data/resume_partial', type=str)
    parser.add_argument('--perturbation_rate', default=0.10, type=float, help='possibility of each kind of perturbation')
    return parser.parse_args()


def generate_labels_dict(entity_types, way='bio'):
    num = 0
    labels_dict = {'O': num}
    num += 1
    for entity_type in entity_types:
        if way == 'bio':
            labels_dict.update({'B-'+entity_type: num})
            num = num + 1
            labels_dict.update({'I-' + entity_type: num})
            num = num + 1
        elif way == 'bioes':
            labels_dict.update({'B-' + entity_type: num})
            num = num + 1
            labels_dict.update({'I-' + entity_type: num})
            num = num + 1
            labels_dict.update({'E-' + entity_type: num})
            num = num + 1
            labels_dict.update({'S-' + entity_type: num})
            num = num + 1
        elif way == 'bmes':
            labels_dict.update({'B-' + entity_type: num})
            num = num + 1
            labels_dict.update({'M-' + entity_type: num})
            num = num + 1
            labels_dict.update({'E-' + entity_type: num})
            num = num + 1
            labels_dict.update({'S-' + entity_type: num})
            num = num + 1
    return labels_dict


def get_entity_types(label_file):
    entity_types = []
    with open(label_file) as f_in:
        for line in f_in.readlines():
            line = line.strip()
            if '-' in line:
                entity_type = line.split('-')[1]
            else:
                continue
            if entity_type not in entity_types:
                entity_types.append(entity_type)
    return entity_types


def extract_entity(data, entity_types, way='bio'):
    for sample in data:
        entities = {}
        for entity_type in entity_types:
            entities.update({entity_type: []})
        length = len(sample['labels'])
        index = 0
        while index < length:
            if way == 'bio':
                if 'B' in sample['labels'][index]:
                    start = index
                    entity_type = sample['labels'][start].split('-')[1]
                    index += 1
                    while index < length and 'I' in sample['labels'][index] and entity_type == sample['labels'][index].split('-')[1]:
                        index += 1
                    end = index - 1
                    entities[entity_type].append([start, end])
                elif 'I' in sample['labels'][index]:
                    index += 1
                    print('error detected!')
                    print(sample['tokens'])
                    print(sample['labels'])
                    print('-'*50)
                elif 'O' in sample['labels'][index]:
                    index += 1
                else:
                    print('This code currently only accept BIO, if you want to use BIOES, BMES or other kind of label, please modify preprocess.py ')
                    print('This code will be stopped soon.')
                    sys.exit()
        sample.update({'entities': entities})


def label_more_or_less(offset):
    start, end = offset
    length = end - start + 1

    # if the length of the entity is less than 3 words, there will be 50% chance to add 1~3 words at each side
    if length < 3:
        if random.uniform(0, 1) > 0.5:
            start -= random.randint(1, 3)
        if random.uniform(0, 1) > 0.5:
            end += random.randint(1, 3)
        return [start, end]

    # There is 30% chance to add 1~6 words, 70% chance to delete 1~2 words
    if random.uniform(0, 1) < 0.3:
        if random.uniform(0, 1) > 0.5:
            start -= random.randint(1, 3)
        if random.uniform(0, 1) > 0.5:
            end += random.randint(1, 3)
    else:
        if random.uniform(0, 1) > 0.5:
            start += 1
        if random.uniform(0, 1) > 0.5:
            end -= 1
    return [start, end]


def divide(offset):
    start, end = offset
    break_point = random.randint(start, end - 1)
    return [start, break_point], [break_point + 1, end]


def mess_with_labels(origin_entity, entity_types):
    probability = 1./float(len(entity_types)-1)
    point = random.uniform(0, 1)
    L = probability
    for entity_type in entity_types:
        if entity_type == origin_entity or entity_type == 'O':
            continue
        if L >= point:
            return entity_type
        else:
            L += probability
    print('something go wrong in function mess_with_labels, this program will be shut down later')
    sys.exit()


def revise_offset(offset, seq_length, occupation):
    # check the boundary
    if offset[0] < 0:
        offset[0] = 0
    if offset[1] >= seq_length:
        offset[1] = seq_length - 1

    new_offsets = []
    # check whether the entity is overlap with other entity. If so, split them.
    index = offset[0]
    while index <= offset[1]:
        if occupation[index] == 0:
            new_start = index
            index += 1
            while index <= offset[1]:
                if occupation[index] == 1:
                    new_end = index - 1
                    new_offsets.append([new_start, new_end])
                    index += 1
                    break
                else:
                    index += 1
        else:
            index += 1
    if occupation[offset[1]] == 0:
        new_offsets.append([new_start, offset[1]])
    return new_offsets


def clear_occupation(offset, occupation):
    for index in range(offset[0], offset[1]+1):
        if occupation[index] == 1:
            occupation[index] = 0
        else:
            print('Something went wrong, please check the code.')
            sys.exit()


def update_occupation(offset, occupation):
    for index in range(offset[0], offset[1]+1):
        if occupation[index] == 0:
            occupation[index] = 1
        else:
            print('Something went wrong, please check the code.')
            sys.exit()


def initial_occupation(sample):
    occupation = [0] * len(sample['tokens'])
    for entity_type in sample['entities']:
        for entity in sample['entities'][entity_type]:
            for index in range(entity[0], entity[1]+1):
                occupation[index] = 1
    return occupation


def perturb_entity(data, perturbation_rate, entity_types, labeler_num=3, spot_check=False, check_proportion=True):
    """
    There could be four kinds of perturbation. Each of them has perturbation_rate chance to happen. They are:
        Add or delete some tokens of an entity to destroy the bound
        Split an entity into two entities
        Change the category of an entity
        Remove an entity from label
    And There is (1 - 4 * perturbation_rate) chance for the entity to be labeled correctly.
    """
    print_flags = False
    entity_count = 0
    mess_count = [0] * 4
    if spot_check:
        if random.uniform(0, 1) > 0.1:
            print_flags = True
    for sample in data:
        new_entities = []
        new_labels = []
        if print_flags:
            print('-' * 50)
            print('before perturb:')
            print(sample['entities'])
        for time in range(labeler_num):
            # perturb entity-level
            occupation = initial_occupation(sample)
            perturbed_entity = {}
            for entity_type in sample['entities']:
                perturbed_entity.update({entity_type: []})
            for entity_type in sample['entities']:
                for entity in sample['entities'][entity_type]:
                    entity_count += 1
                    point = random.uniform(0, 1)
                    if 0 <= point < perturbation_rate:
                        mess_count[0] += 1
                        clear_occupation(entity, occupation)
                        entity = label_more_or_less(entity)
                        for result in revise_offset(entity, len(sample['tokens']), occupation):
                            perturbed_entity[entity_type].append(result)
                            update_occupation(result, occupation)
                    elif perturbation_rate <= point < 2 * perturbation_rate:
                        mess_count[1] += 1
                        if entity[0] != entity[1]:
                            entity, new_entity = divide(entity)
                            perturbed_entity[entity_type].append(entity)
                            perturbed_entity[entity_type].append(new_entity)
                        else:
                            perturbed_entity[entity_type].append(entity)
                    elif 2 * perturbation_rate <= point < 3 * perturbation_rate:
                        mess_count[2] += 1
                        new_entity_type = mess_with_labels(entity_type, entity_types)
                        perturbed_entity[new_entity_type].append(entity)
                    elif 3 * perturbation_rate <= point < 4 * perturbation_rate:
                        mess_count[3] += 1
                    else:
                        perturbed_entity[entity_type].append(entity)    # stay the same
            new_entities.append(perturbed_entity)
            # synchronize token-level
            perturbed_labels = ['O'] * len(sample['tokens'])
            for entity_type in perturbed_entity:
                for start, end in perturbed_entity[entity_type]:
                    perturbed_labels[start] = 'B-'+entity_type
                    if start != end:
                        for index in range(start+1, end+1):
                            perturbed_labels[index] = 'I-'+entity_type
            new_labels.append(perturbed_labels)
        sample['entities'] = new_entities
        sample['labels'] = new_labels
        if print_flags:
            print('after perturb:')
            for index in range(len(sample['entities'])):
                print(sample['entities'][index])
                print(sample['labels'][index])
    if check_proportion:
        print('have {} entity'.format(entity_count))
        for index, count in enumerate(mess_count):
            print('perturb {} entity in way {}'.format(count/entity_count, index+1))


def restore_format(data):
    for sample in data:
        sample.pop('entities')


def perturb(args, entity_types):
    assert args.perturbation_rate <= 0.25, 'perturbation rate should be less than 0.25 because there are four kinds of perturbation'
    file = os.path.join(args.output_dir, 'train.json')
    with open(file, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)

    # process data
    extract_entity(data, entity_types)
    perturb_entity(data, args.perturbation_rate, entity_types)
    restore_format(data)

    # output result
    with open(os.path.join(file), 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=2)


def modify_format(args):
    dataset_list = ['dev', 'test', 'train']

    for subset in dataset_list:
        raw_file = os.path.join(args.raw_data_dir, subset+'.txt')
        output_file = os.path.join(args.output_dir, subset+'.json')
        if os.path.exists(raw_file):
            with open(raw_file, 'r', encoding='utf-8') as f_in:
                lines = f_in.readlines()
                data = []
                tokens = []
                labels = []
                for line in lines:
                    if len(line) > 1:  # more than just '\n'
                        token, label = line.split(' ')
                        tokens.append(token)
                        labels.append(label.strip())
                    elif len(tokens) > 1 and len(labels) > 1:
                        data.append({
                            'tokens': tokens,
                            'labels': labels
                        })
                        tokens = []
                        labels = []
            with open(output_file, 'w', encoding='utf-8') as f_out:
                json.dump(data, f_out, ensure_ascii=False, indent=2)
        else:
            print('No {} set detected. Skip.'.format(subset))

    label_file = os.path.join(args.raw_data_dir, 'labels.txt')
    assert os.path.exists(label_file), 'There is no {}, please check it out.'.format(label_file)
    ent2id_file = os.path.join(args.output_dir, 'bio_ent2id.json')
    entity_types = get_entity_types(label_file)
    ent2id = generate_labels_dict(entity_types)
    with open(ent2id_file, 'w', encoding='utf-8') as f_out:
        json.dump(ent2id, f_out, ensure_ascii=False, indent=2)
    return entity_types


if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    entity_types = modify_format(args)
    perturb(args, entity_types)
