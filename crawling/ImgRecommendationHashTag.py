import os
import pandas as pd
from collections import Counter


def delete_unrelated_tag(index, total_most_used_tags):
    for idx1 in range(len(total_most_used_tags)):
        if index != idx1:
            for item in total_most_used_tags[index][:]:
                if item in total_most_used_tags[idx1]:
                    print(item + " 이(가) 삭제되었습니다.")
                    total_most_used_tags[index].remove(item)
    return total_most_used_tags[index]


def delete_common_tag(database, renew_database):
    renew_total_tag_list = []
    for idx1 in range(len(database)):
        table = []
        total_tag_list = []
        for idx2 in range(len(database[idx1])):
            tuple = []
            for idx3 in range(len(database[idx1][idx2])):
                tag_list = []
                for idx4 in range(len(database[idx1][idx2][idx3])):
                    if database[idx1][idx2][idx3][idx4] not in stop_word:
                        tag_list.append(database[idx1][idx2][idx3][idx4])
                        total_tag_list.append(database[idx1][idx2][idx3][idx4])
                tuple.append(tag_list)
            table.append(tuple)
        renew_database.append(table)
        renew_total_tag_list.append(total_tag_list)
    return renew_database, renew_total_tag_list


def get_tag_from_csv(csv_path, words, stop_words):
    database = []
    for index in range(len(words)):
        # csv 파일 읽기
        csv = pd.read_csv(csv_path + '{}.csv'.format(words[index]),
                          header=0,
                          names=['index', 'date', 'like', 'place', 'tags'],
                          usecols=['tags'])
        # csv 파일 내용 배열화
        table = []
        for row in range(csv.shape[0]):
            tuple = []
            for col in range(csv.shape[1]):
                # 해당 튜플값의 정보가 없는 경우
                if type(csv.loc[row][col]) is not str:
                    tuple.append('')
                # 해당 튜플값의 정보가 있는 경우
                else:
                    tags = csv.loc[row][col].strip('[]')
                    tags = tags.replace(",", "")
                    tags = tags.replace("'", "")
                    tag_list = tags.split(' ')
                    for idx in range(len(tag_list)):
                        if tag_list[idx] not in stop_words:
                            total_tag_list.append(tag_list[idx])
                    # print("{}.csv파일의 {}번째 {} : {}".format(words[index], row + 1, csv.columns[col], tag_list))
                    tuple.append(tag_list)
            table.append(tuple)
        database.append(table)
    return database


def find_csv_file(csv_path):
    csv_files = []
    for file in os.listdir(csv_path):
        file_name = file.split('.')
        if file_name[1] == "csv" and file_name[0][0] == "#":
            csv_files.append(file_name[0])
    return csv_files


if __name__ == '__main__':
    # csv파일 경로
    csv_path = './csv_file/'
    # 검색태그 리스트
    csv_files = find_csv_file(csv_path)
    # 불용어
    stop_word = []
    # 모든 csv 파일 내 태그 배열
    total_tag_list = []
    # 전체 테이블 배열
    database = get_tag_from_csv(csv_path, csv_files, csv_files)
    print("데이터베이스 내 테이블의 갯수 : {}개\n".format(len(database)))
    for idx1 in range(len(database)):
        print("{} 테이블 내 튜플의 갯수 : {}개".format(csv_files[idx1], len(database[idx1])))
        num = 0
        for idx2 in range(len(database[idx1])):
            num = num + len(database[idx1][idx2][0])
        print("{} 테이블 내 태그의 갯수 : {}개\n".format(csv_files[idx1], num))

    print("데이터베이스 내 태그의 갯수 : {}개".format(len(total_tag_list)))

    count = Counter(total_tag_list)
    most_used_tag_list = count.most_common(50)

    most_used_tags = []
    most_used_tags_count = []

    for idx in range(len(most_used_tag_list)):
        most_used_tags.append(most_used_tag_list[idx][0])
        most_used_tags_count.append(most_used_tag_list[idx][1])

    data = {
        'tag': most_used_tags,
        'count': most_used_tags_count
    }
    frame = pd.DataFrame(data)
    frame.to_csv(csv_path + "most_used_tag.csv", encoding='utf-8-sig')

    for idx in range(len(most_used_tags)):
        if most_used_tags[idx] not in stop_word:
            stop_word.append(most_used_tags[idx])
    for idx in range(len(csv_files)):
        if csv_files[idx] not in stop_word:
            stop_word.append(csv_files[idx])
    print("불용어 갯수 : {}개".format(len(stop_word)))
    print(stop_word)

    renew_database = []
    renew_database, renew_total_tag_list = delete_common_tag(database, renew_database)
    renew_csv_path = 'Top50/'

    total_most_used_tags = []
    print("데이터베이스 내 테이블의 갯수 : {}개\n".format(len(renew_database)))
    for idx1 in range(len(renew_database)):
        print("{} 테이블 내 튜플의 갯수 : {}개".format(csv_files[idx1], len(renew_database[idx1])))
        print("{} 테이블 내 태그의 갯수 : {}개\n".format(csv_files[idx1], len(renew_total_tag_list[idx1])))

        count = Counter(renew_total_tag_list[idx1])
        most_used_tag_list = count.most_common(50)

        most_used_tags = []
        most_used_tags_count = []

        for idx in range(len(most_used_tag_list)):
            most_used_tags.append(most_used_tag_list[idx][0])
            most_used_tags_count.append(most_used_tag_list[idx][1])

        data = {
            'tag': most_used_tags,
            'count': most_used_tags_count
        }
        frame = pd.DataFrame(data)
        frame.to_csv(renew_csv_path + csv_files[idx1] + "_Top_50.csv", encoding='utf-8-sig')
        total_most_used_tags.append(most_used_tags)

    for index in range(len(total_most_used_tags)):
        print("{} 테이블 내 TOP 50 해시태그 갯수 : {}개".format(csv_files[index], len(total_most_used_tags[index])))
        final_hashtag = delete_unrelated_tag(index, total_most_used_tags)
        print("{} 테이블 내 최종 해시태그 갯수 : {}개".format(csv_files[index], len(final_hashtag)))
        data = {
            'tag': final_hashtag[:10],
        }
        frame = pd.DataFrame(data)
        frame.to_csv("./recommended_hashtag/" + csv_files[index] + "_Top_10.csv", encoding='utf-8-sig')
        print("")
